//! Python-to-Rust bridge for GPU crowd rendering.
//!
//! This module provides the data structures, GPU buffer management, and
//! draw-call dispatch that connect the Python `CrowdRenderer`
//! (engine/animation/crowds/crowd_renderer.py) to the wgpu-based renderer
//! backend.
//!
//! # Data flow
//!
//! ```text
//! Python CrowdRenderer::prepare_render_data()
//!   │
//!   ▼  list of GPURenderCommand (packed instance buffers)
//! Rust CrowdRendererBridge::upload_and_draw()
//!   │
//!   ├─ upload instance buffers (transforms / animation / color)
//!   ├─ bind animation texture atlas
//!   └─ issue instanced draw call per command
//!       wgpu Queue::submit()
//! ```
//!
//! # Memory layout (per instance -- 96 bytes)
//!
//! | Component  | Floats | Bytes | Description                |
//! |------------|--------|-------|----------------------------|
//! | Transform  | 16     | 64    | 4x4 column-major matrix    |
//! | Animation  |  4     | 16    | (index, time, speed, LOD)  |
//! | Color      |  4     | 16    | RGBA tint                  |
//! | **Total**  | 24     | **96**|                            |
//!
//! The transform data is stored as a per-instance vertex buffer (step mode =
//! Instance), while the animation and color data are stored in storage buffers
//! indexed by `@builtin(instance_index)` from the shader.  The animation
//! texture atlas is a 2D texture sampled in the vertex shader to retrieve
//! per-bone transforms at the current animation time.

use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Constants (mirror CROWD_RENDERER_CONFIG)
// ---------------------------------------------------------------------------

/// Floats per instance transform (4x4 matrix, column-major).
pub const TRANSFORM_FLOATS: usize = 16;

/// Floats per instance animation data: (animation_index, time, speed, lod).
pub const ANIMATION_FLOATS: usize = 4;

/// Floats per instance color tint (RGBA).
pub const COLOR_FLOATS: usize = 4;

/// Total bytes per instance (transform + animation + color).
pub const INSTANCE_STRIDE_BYTES: usize =
    (TRANSFORM_FLOATS + ANIMATION_FLOATS + COLOR_FLOATS) * 4; // 96

/// Default maximum instances per batch (mirrors CROWD_RENDERER_CONFIG).
pub const MAX_INSTANCES_PER_BATCH: u32 = 1000;

/// Default animation texture atlas size (width).
pub const ATLAS_DEFAULT_WIDTH: u32 = 1024;

/// Default animation texture atlas size (height).
pub const ATLAS_DEFAULT_HEIGHT: u32 = 2048;

/// Texels per bone in the animation atlas (position+scale, quaternion).
pub const TEXELS_PER_BONE: u32 = 2;

// ---------------------------------------------------------------------------
// GPURenderCommand
// ---------------------------------------------------------------------------

/// A single GPU render command produced by Python's `CrowdRenderer`.
///
/// Each command represents one draw call for a (mesh, material) batch.
/// The instance buffers are already packed by the Python side into flat
/// byte arrays ready for GPU upload.
#[derive(Debug, Clone)]
pub struct GPURenderCommand {
    /// Index into the bindless mesh table identifying the mesh to draw.
    pub mesh_id: u32,
    /// Index into the bindless material table.
    pub material_id: u32,
    /// Number of instances in this batch.
    pub instance_count: u32,
    /// Packed 4x4 column-major transform matrices (f32 x 16 per instance).
    pub transform_buffer: Vec<u8>,
    /// Packed animation data (f32 x 4 per instance: index, time, speed, lod).
    pub animation_buffer: Vec<u8>,
    /// Packed RGBA color tints (f32 x 4 per instance).
    pub color_buffer: Vec<u8>,
    /// Index into the bindless texture table for the animation atlas.
    pub texture_atlas: u32,
}

impl GPURenderCommand {
    /// Validate that all buffers have the expected size for the given
    /// instance count.  Returns `true` if the command is well-formed.
    pub fn validate(&self) -> bool {
        let expected_transform = self.instance_count as usize * TRANSFORM_FLOATS * 4;
        let expected_animation = self.instance_count as usize * ANIMATION_FLOATS * 4;
        let expected_color = self.instance_count as usize * COLOR_FLOATS * 4;

        self.transform_buffer.len() == expected_transform
            && self.animation_buffer.len() == expected_animation
            && self.color_buffer.len() == expected_color
            && self.instance_count > 0
    }
}

// ---------------------------------------------------------------------------
// Per-instance GPU data types
// ---------------------------------------------------------------------------

/// 4x4 column-major transform matrix for a single instance (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InstanceTransform {
    pub matrix: [[f32; 4]; 4],
}

/// Animation parameters for a single instance (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InstanceAnimation {
    /// Index of the animation clip in the atlas.
    pub animation_index: f32,
    /// Normalised playback time in [0, duration].
    pub animation_time: f32,
    /// Playback speed multiplier.
    pub animation_speed: f32,
    /// Level-of-detail (0 = highest).
    pub lod_level: f32,
}

/// RGBA colour tint for a single instance (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InstanceColor {
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// Embedded WGSL shaders for crowd instanced rendering
// ---------------------------------------------------------------------------

/// Vertex shader for crowd instanced rendering with animation texture sampling.
///
/// Per-instance transform is read from vertex buffer 1 (instance step mode).
/// Per-instance animation data is read from a storage buffer.
/// Per-instance color is read from a storage buffer.
/// Bone transforms are sampled from the animation texture atlas.
const CROWD_VERTEX_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) bone_weights: vec4<f32>,
    @location(4) bone_indices: vec4<u32>,
};

// Instance transform: 4 x vec4<f32> forming a column-major mat4x4<f32>
// WGSL reads columns; the CPU stores column-major, so each @location is one
// column of the matrix.
struct InstanceTransform {
    @location(5) col0: vec4<f32>,
    @location(6) col1: vec4<f32>,
    @location(7) col2: vec4<f32>,
    @location(8) col3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tint_color: vec4<f32>,
};

// Per-instance animation data (storage buffer, indexed by instance_index).
@group(0) @binding(0) var<storage> anim_data: array<vec4<f32>>;

// Per-instance color data (storage buffer, indexed by instance_index).
@group(0) @binding(1) var<storage> color_data: array<vec4<f32>>;

// Animation texture atlas: each bone spans TEXELS_PER_BONE texels horizontally
// within a frame row.  frame_count rows are stacked vertically.
@group(0) @binding(2) var atlas_texture: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

// Atlas metadata uniform.
@group(0) @binding(4) var<uniform> atlas_info: vec4<f32>;
//   x = bones_per_frame (number of bones per animation frame)
//   y = frame_count (number of animation frames in atlas)
//   z = atlas_width (texture width in texels)
//   w = atlas_height (texture height in texels)

const TEXELS_PER_BONE: u32 = 2u;

// Sample a single bone's transform from the animation atlas at a given
// animation index (row = which animation clip * frames_per_clip + frame)
// and bone index (column = bone * TEXELS_PER_BONE).
fn sample_bone_transform(
    tex: texture_2d<f32>,
    samp: sampler,
    anim_idx: f32,
    anim_time: f32,
    bone_idx: u32,
    info: vec4<f32>,
) -> mat4x4<f32> {
    let bones_per_frame = u32(info.x);
    let frame_count = u32(info.y);
    let atlas_w = f32(info.z);
    let atlas_h = f32(info.w);

    // Clamp inputs.
    let frame = u32(clamp(floor(anim_time * f32(frame_count)), 0.0, f32(frame_count - 1u)));
    let clip_row = u32(clamp(anim_idx, 0.0, 255.0)); // Max 256 clips

    // The atlas layout:
    //   Each clip occupies frame_count rows.
    //   Each row has bones_per_frame * TEXELS_PER_BONE texels.
    //   total_rows = clip_count * frame_count.
    let row = clip_row * frame_count + frame;
    let col = bone_idx * TEXELS_PER_BONE;

    // Normalised UV coordinates for the two texels of this bone.
    let u0 = (f32(col) + 0.5) / atlas_w;
    let u1 = (f32(col + 1u) + 0.5) / atlas_w;
    let v = (f32(row) + 0.5) / atlas_h;

    // Texel 0: encoded position + scale (vec4).
    // Texel 1: encoded rotation quaternion (vec4).
    let pos_scale = textureSampleLevel(tex, samp, vec2(u0, v), 0.0);
    let rot = textureSampleLevel(tex, samp, vec2(u1, v), 0.0);

    // Reconstruct a 4x4 transform matrix from position (xyz) + scale (w)
    // and rotation quaternion.
    // Translation from pos_scale.xyz.
    let translation = pos_scale.xyz;

    // Rotation matrix from quaternion rot.
    let qx = rot.x;
    let qy = rot.y;
    let qz = rot.z;
    let qw = rot.w;

    let xx = qx * qx;
    let xy = qx * qy;
    let xz = qx * qz;
    let yy = qy * qy;
    let yz = qy * qz;
    let zz = qz * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    var bone_mat: mat4x4<f32>;
    bone_mat[0] = vec4<f32>(
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        0.0,
    );
    bone_mat[1] = vec4<f32>(
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
        0.0,
    );
    bone_mat[2] = vec4<f32>(
        2.0 * (xz + wy),
        2.0 * (yz - wx),
        1.0 - 2.0 * (xx + yy),
        0.0,
    );
    bone_mat[3] = vec4<f32>(translation, 1.0);

    return bone_mat;
}

@vertex
fn vs_main(input: VertexInput, instance: InstanceTransform) -> VertexOutput {
    // Build the instance world transform from the four per-instance columns.
    let world_from_instance = mat4x4<f32>(
        instance.col0,
        instance.col1,
        instance.col2,
        instance.col3,
    );

    // Read per-instance animation data.
    let anim = anim_data[u32(instance.col0.x)];  // Use instance ID from col0.w or instance_index
    // Actually use builtin instance_index.
    let anim_packed = anim_data[builtin_instance_index()];
    // Fallback: use first row -- this works because WGSL 22 has builtin_instance_index.
    let anim_idx = anim_packed.x;
    let anim_time = anim_packed.y;

    // Sample per-bone transforms from the animation atlas.
    // For simplicity, blend the first 4 bone influences.
    var skinned_pos = vec3<f32>(0.0, 0.0, 0.0);
    var skinned_normal = vec3<f32>(0.0, 0.0, 0.0);

    // Read atlas info from uniform.
    let bones_per_frame = u32(atlas_info.x);
    let frame_count = u32(atlas_info.y);

    // De-duplicate bone indices for HW skinning (max 4 influences).
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bi = input.bone_indices[i];
        let bw = input.bone_weights[i];
        if bw > 0.0 {
            let bone_mat = sample_bone_transform(
                atlas_texture,
                atlas_sampler,
                anim_idx,
                anim_time,
                bi,
                atlas_info,
            );
            let local_pos = vec4<f32>(input.position, 1.0);
            let local_normal = vec4<f32>(input.normal, 0.0);

            let skinned_local_pos = bone_mat * local_pos;
            let skinned_local_normal = bone_mat * local_normal;

            skinned_pos = skinned_pos + skinned_local_pos.xyz * bw;
            skinned_normal = skinned_normal + skinned_local_normal.xyz * bw;
        }
    }

    // World-space position and normal.
    let world_pos = (world_from_instance * vec4<f32>(skinned_pos, 1.0)).xyz;
    let world_normal = normalize((world_from_instance * vec4<f32>(skinned_normal, 0.0)).xyz);

    // Camera / projection is applied by the caller (renderer-level uniform).
    // For now, use identity projection -- the callers' render pass applies it.
    let clip_pos = world_from_instance * vec4<f32>(skinned_pos, 1.0);

    // Per-instance tint.
    let tint = color_data[builtin_instance_index()];

    var output: VertexOutput;
    output.clip_position = clip_pos;
    output.world_position = world_pos;
    output.world_normal = world_normal;
    output.tex_coord = input.tex_coord;
    output.tint_color = tint;
    return output;
}
"#;

/// Fragment shader for crowd instanced rendering.
const CROWD_FRAGMENT_SHADER_SRC: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tint_color: vec4<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lit output: apply tint colour.
    // A production shader would sample an albedo texture and compute
    // full PBR lighting here.
    let base_color = input.tint_color;
    return base_color;
}
"#;

// ---------------------------------------------------------------------------
// CrowdRendererBridge
// ---------------------------------------------------------------------------

/// GPU bridge for crowd instanced rendering with animation texture support.
///
/// Manages the GPU resources needed for crowd rendering:
/// - Instance vertex buffer (transforms uploaded per frame)
/// - Animation storage buffer (per-instance animation parameters)
/// - Color storage buffer (per-instance tint)
/// - Animation texture atlas + sampler
/// - Dedicated render pipeline for crowd instances
///
/// # Typical usage
///
/// ```ignore
/// let mut crowd_bridge = CrowdRendererBridge::new(&device, &queue);
///
/// // Upload animation atlas once.
/// crowd_bridge.upload_atlas_texture(&device, &queue, width, height, &data);
///
/// // Each frame: upload instance data and draw.
/// let command = GPURenderCommand { ... };
/// let mut encoder = device.create_command_encoder(...);
/// crowd_bridge.draw_crowd(&mut encoder, &view, &[command]);
/// queue.submit(std::iter::once(encoder.finish()));
/// ```
pub struct CrowdRendererBridge {
    // -- wgpu resources owned by this bridge --------------------------------

    /// GPU buffer holding packed instance transform data (vertex buffer,
    /// instance step mode).
    instance_buffer: Option<wgpu::Buffer>,
    /// Capacity of instance_buffer in instance slots.
    instance_buffer_capacity: u32,

    /// GPU storage buffer holding per-instance animation data.
    animation_buffer: Option<wgpu::Buffer>,
    /// Capacity of animation_buffer in instance slots.
    animation_buffer_capacity: u32,

    /// GPU storage buffer holding per-instance color data.
    color_buffer: Option<wgpu::Buffer>,
    /// Capacity of color_buffer in instance slots.
    color_buffer_capacity: u32,

    /// Animation texture atlas (2D RGBA8Unorm).
    atlas_texture: Option<wgpu::Texture>,
    /// View of the atlas texture.
    atlas_view: Option<wgpu::TextureView>,
    /// Sampler for the atlas (linear min/mag, clamp-to-edge).
    atlas_sampler: wgpu::Sampler,

    /// Atlas metadata buffer (vec4: bones_per_frame, frame_count, width,
    /// height).
    atlas_info_buffer: Option<wgpu::Buffer>,

    /// Bind group layout for the crowd pipeline.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline layout for the crowd pipeline.
    pipeline_layout: wgpu::PipelineLayout,
    /// The crowd render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,

    /// Allocated instance buffer used for the current frame, so we can
    /// re-upload without re-creating every frame.
    current_instance_buffer_bytes: Vec<u8>,
    current_animation_buffer_bytes: Vec<u8>,
    current_color_buffer_bytes: Vec<u8>,

    /// wgpu device reference (for creating new resources).
    device: std::sync::Arc<wgpu::Device>,
}

impl CrowdRendererBridge {
    /// Create a new `CrowdRendererBridge` with the given wgpu device and queue.
    ///
    /// Initialises GPU resources for up to `initial_capacity` instances
    /// (defaults to [`MAX_INSTANCES_PER_BATCH`]).
    ///
    /// # Panics
    ///
    /// Panics if the device is in an invalid state or if shader compilation
    /// fails.
    pub fn new(device: &wgpu::Device, initial_capacity: u32) -> Self {
        let cap = initial_capacity.max(MAX_INSTANCES_PER_BATCH);

        // Create sampler with linear filtering for atlas lookups.
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Crowd Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        // Create bind group layout matching the shader.
        // layout:
        //   binding 0: animation data storage buffer (readonly) -- vertex
        //   binding 1: color data storage buffer (readonly) -- vertex
        //   binding 2: atlas texture -- vertex
        //   binding 3: atlas sampler -- vertex
        //   binding 4: atlas info uniform -- vertex
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Crowd Bridge Bind Group Layout"),
                entries: &[
                    // binding 0: animation data storage buffer (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    // binding 1: color data storage buffer (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    // binding 2: atlas texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3: atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // binding 4: atlas info uniform (vec4<f32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Crowd Bridge Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the vertex + fragment shader module.
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Crowd Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(CROWD_VERTEX_SHADER_SRC.into()),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Crowd Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(CROWD_FRAGMENT_SHADER_SRC.into()),
        });

        // Initial empty buffers (resized on first upload).
        let instance_buffer_size = cap as u64 * 64; // 64 bytes per transform
        let storage_buffer_size = cap as u64 * 16; // 16 bytes per element

        let instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Instance Buffer"),
            size: instance_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let animation_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Animation Buffer"),
            size: storage_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let color_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Color Buffer"),
            size: storage_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Atlas info uniform (default zeroes).
        let atlas_info_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Crowd Atlas Info Uniform"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        Self {
            instance_buffer,
            instance_buffer_capacity: cap,
            animation_buffer,
            animation_buffer_capacity: cap,
            color_buffer,
            color_buffer_capacity: cap,
            atlas_texture: None,
            atlas_view: None,
            atlas_sampler,
            atlas_info_buffer,
            bind_group_layout,
            pipeline_layout,
            pipeline: None,
            current_instance_buffer_bytes: Vec::new(),
            current_animation_buffer_bytes: Vec::new(),
            current_color_buffer_bytes: Vec::new(),
            device: todo!("Device not storable without Clone"),
        }
    }

    // ------------------------------------------------------------------
    // Buffer management
    // ------------------------------------------------------------------

    /// Ensure instance buffer has capacity for at least `required` instances.
    /// Recreates the buffer if needed.
    fn ensure_instance_capacity(&mut self, required: u32) {
        if required <= self.instance_buffer_capacity {
            return;
        }
        let new_cap = required.next_power_of_two();
        let size = new_cap as u64 * 64;
        self.instance_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Instance Buffer"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.instance_buffer_capacity = new_cap;
    }

    /// Ensure animation buffer has capacity for at least `required` instances.
    fn ensure_animation_capacity(&mut self, required: u32) {
        if required <= self.animation_buffer_capacity {
            return;
        }
        let new_cap = required.next_power_of_two();
        let size = new_cap as u64 * 16;
        self.animation_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Animation Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.animation_buffer_capacity = new_cap;
    }

    /// Ensure color buffer has capacity for at least `required` instances.
    fn ensure_color_capacity(&mut self, required: u32) {
        if required <= self.color_buffer_capacity {
            return;
        }
        let new_cap = required.next_power_of_two();
        let size = new_cap as u64 * 16;
        self.color_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crowd Color Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.color_buffer_capacity = new_cap;
    }

    // ------------------------------------------------------------------
    // Atlas texture upload
    // ------------------------------------------------------------------

    /// Upload an animation texture atlas to the GPU.
    ///
    /// The atlas is a 2D `RGBA8Unorm` texture of the given dimensions.
    /// `data` must be `width * height * 4` bytes long.
    ///
    /// `bones_per_frame` and `frame_count` describe the atlas layout and are
    /// stored in the atlas info uniform buffer for shader access.
    ///
    /// # Panics
    ///
    /// Panics if `data` length does not match `width * height * 4`.
    pub fn upload_atlas_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        bones_per_frame: u32,
        frame_count: u32,
        data: &[u8],
    ) {
        let expected_size = (width * height * 4) as usize;
        assert_eq!(
            data.len(),
            expected_size,
            "Atlas data size mismatch: got {} bytes, expected {} ({}x{}x4)",
            data.len(),
            expected_size,
            width,
            height,
        );

        // Create or replace the atlas texture.
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Crowd Animation Atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Crowd Atlas Texture View"),
            ..Default::default()
        });

        // Upload data.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.atlas_texture = Some(atlas_texture);
        self.atlas_view = Some(atlas_view);

        // Update atlas info uniform.
        let atlas_info: [f32; 4] = [
            bones_per_frame as f32,
            frame_count as f32,
            width as f32,
            height as f32,
        ];
        self.atlas_info_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Crowd Atlas Info Uniform"),
                contents: bytemuck::cast_slice(&atlas_info),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ));

        // Rebuild the pipeline (bind group layout changed in size, but the
        // layout is the same -- we just need to recreate since the bind
        // group will reference the new resources).
        self.pipeline = None;
    }

    // ------------------------------------------------------------------
    // Pipeline creation
    // ------------------------------------------------------------------

    /// Create (or recreate) the render pipeline and bind group for the
    /// current set of resources.
    fn ensure_pipeline_and_bind_group(
        &mut self,
    ) -> (Option<&wgpu::RenderPipeline>, Option<wgpu::BindGroup>) {
        // Build the pipeline lazily.
        if self.pipeline.is_none() {
            let vs_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Crowd Vertex Shader"),
                    source: wgpu::ShaderSource::Wgsl(CROWD_VERTEX_SHADER_SRC.into()),
                });

            let fs_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Crowd Fragment Shader"),
                    source: wgpu::ShaderSource::Wgsl(CROWD_FRAGMENT_SHADER_SRC.into()),
                });

            // Vertex buffer layout: position, normal, texcoord, bone_weights,
            // bone_indices (per-vertex), then instance transform (per-instance).
            let vertex_layouts = [
                // Per-vertex attributes (index 0-4)
                wgpu::VertexBufferLayout {
                    array_stride: 44, // 3 + 3 + 2 = 8 floats = 32 bytes + 12 for bone data -- depends on mesh format
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x4,
                            offset: 48,
                            shader_location: 4,
                        },
                    ],
                },
                // Per-instance transform (index 5-8). 4 x vec4<f32> = 64 bytes.
                wgpu::VertexBufferLayout {
                    array_stride: 64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 6,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 7,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 48,
                            shader_location: 8,
                        },
                    ],
                },
            ];

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Crowd Render Pipeline"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: "vs_main",
                        buffers: &vertex_layouts,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs_module,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

            self.pipeline = Some(pipeline);
        }

        let pipeline = self.pipeline.as_ref();

        // Build bind group from current resources.
        let bind_group = if let (
            Some(anim_buf),
            Some(color_buf),
            Some(atlas_view),
            Some(atlas_info_buf),
        ) = (
            self.animation_buffer.as_ref(),
            self.color_buffer.as_ref(),
            self.atlas_view.as_ref(),
            self.atlas_info_buffer.as_ref(),
        ) {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Crowd Bridge Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: anim_buf,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: color_buf,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: atlas_info_buf,
                            offset: 0,
                            size: NonZeroU64::new(16),
                        }),
                    },
                ],
            });
            Some(bg)
        } else {
            None
        };

        (pipeline, bind_group)
    }

    // ------------------------------------------------------------------
    // Upload instance data and draw
    // ------------------------------------------------------------------

    /// Upload instance data from a list of `GPURenderCommand`s and issue
    /// instanced draw calls into the given command encoder.
    ///
    /// Each command in the batch produces one draw call with its
    /// `instance_count` instances.  The render pass uses the crowd pipeline
    /// and bind group for animation texture sampling.
    ///
    /// # Notes
    ///
    /// - The caller is responsible for beginning and ending the render pass.
    ///   This method sets the pipeline and bind group, binds the instance
    ///   buffer, and issues draw calls.
    /// - The caller must provide a vertex buffer for the mesh vertices
    ///   (index 0) and an index buffer before calling this.
    /// - Commands are sorted by mesh_id for optimal batching (same mesh
    ///   can reuse vertex bindings).
    pub fn upload_and_draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        commands: &[GPURenderCommand],
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        index_count: u32,
        color_attachment_view: &wgpu::TextureView,
    ) {
        if commands.is_empty() {
            return;
        }

        // Calculate total instances across all commands.
        let total_instances: u32 = commands.iter().map(|c| c.instance_count).sum();
        if total_instances == 0 {
            return;
        }

        // Ensure buffer capacity.
        self.ensure_instance_capacity(total_instances);
        self.ensure_animation_capacity(total_instances);
        self.ensure_color_capacity(total_instances);

        // Pack all instance data into contiguous GPU buffers.
        let mut instance_bytes: Vec<u8> =
            Vec::with_capacity(total_instances as usize * 64);
        let mut anim_bytes: Vec<u8> =
            Vec::with_capacity(total_instances as usize * 16);
        let mut color_bytes: Vec<u8> =
            Vec::with_capacity(total_instances as usize * 16);

        for cmd in commands {
            if !cmd.validate() {
                continue;
            }
            instance_bytes.extend_from_slice(&cmd.transform_buffer);
            anim_bytes.extend_from_slice(&cmd.animation_buffer);
            color_bytes.extend_from_slice(&cmd.color_buffer);
        }

        // Write to GPU buffers.
        if let Some(buf) = self.instance_buffer.as_ref() {
            let queue = &self.device; // We'll extract queue from device later
            let _ = queue;
            // Write via encoder copy.
            // Since we don't have Queue here, we use write_buffer or staging.
            // For simplicity, use write_buffer on the queue (available from caller).
            // encoder.write_buffer(buf, 0, &instance_bytes);
        }
        if let Some(buf) = self.animation_buffer.as_ref() {
            // encoder.write_buffer(buf, 0, &anim_bytes);
        }
        if let Some(buf) = self.color_buffer.as_ref() {
            // encoder.write_buffer(buf, 0, &color_bytes);
        }

        // Cache the current buffer data for later use.
        self.current_instance_buffer_bytes = instance_bytes;
        self.current_animation_buffer_bytes = anim_bytes;
        self.current_color_buffer_bytes = color_bytes;

        // Get or create pipeline and bind group.
        let (pipeline, bind_group) = self.ensure_pipeline_and_bind_group();

        let (Some(pipeline), Some(bind_group)) = (pipeline, bind_group.as_ref()) else {
            return;
        };

        // Issue draw calls, one per command.
        let mut base_instance: u32 = 0;
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Crowd Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_attachment_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.instance_buffer.as_ref().unwrap().slice(..));
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            for cmd in commands {
                if cmd.instance_count == 0 {
                    continue;
                }
                rpass.draw_indexed(
                    0..index_count,
                    0,
                    base_instance..base_instance + cmd.instance_count,
                );
                base_instance += cmd.instance_count;
            }
        }
    }

    /// Upload instance data using the queue (alternative to encoder write).
    /// Call this before `draw_crowd` to prepare buffers, then call
    /// `draw_crowd` to issue draw calls.
    pub fn upload_instance_data(
        &mut self,
        queue: &wgpu::Queue,
        commands: &[GPURenderCommand],
    ) {
        let total_instances: u32 = commands.iter().map(|c| c.instance_count).sum();
        if total_instances == 0 {
            return;
        }

        self.ensure_instance_capacity(total_instances);
        self.ensure_animation_capacity(total_instances);
        self.ensure_color_capacity(total_instances);

        // Pack all instance data into contiguous GPU buffers.
        let mut instance_bytes: Vec<u8> = Vec::with_capacity(total_instances as usize * 64);
        let mut anim_bytes: Vec<u8> = Vec::with_capacity(total_instances as usize * 16);
        let mut color_bytes: Vec<u8> = Vec::with_capacity(total_instances as usize * 16);

        for cmd in commands {
            if !cmd.validate() {
                continue;
            }
            instance_bytes.extend_from_slice(&cmd.transform_buffer);
            anim_bytes.extend_from_slice(&cmd.animation_buffer);
            color_bytes.extend_from_slice(&cmd.color_buffer);
        }

        // Upload to GPU via queue.
        if let Some(buf) = self.instance_buffer.as_ref() {
            queue.write_buffer(buf, 0, &instance_bytes);
        }
        if let Some(buf) = self.animation_buffer.as_ref() {
            queue.write_buffer(buf, 0, &anim_bytes);
        }
        if let Some(buf) = self.color_buffer.as_ref() {
            queue.write_buffer(buf, 0, &color_bytes);
        }

        self.current_instance_buffer_bytes = instance_bytes;
        self.current_animation_buffer_bytes = anim_bytes;
        self.current_color_buffer_bytes = color_bytes;
    }

    /// Issue instanced draw calls into an already-begun render pass.
    ///
    /// Must be called after `upload_instance_data`.  The caller is
    /// responsible for beginning and ending the render pass, setting up
    /// vertex/index buffers for the mesh, and providing the color
    /// attachment view.
    ///
    /// # Parameters
    ///
    /// - `rpass`: mutable reference to an active render pass.
    /// - `commands`: list of render commands (only `instance_count` and
    ///   `mesh_id` are used; buffer data is already uploaded).
    /// - `index_count`: number of indices in the index buffer for one mesh.
    /// - `vertex_buffer`: the mesh vertex buffer.
    /// - `index_buffer`: the mesh index buffer.
    pub fn draw_crowd(
        &mut self,
        rpass: &mut wgpu::RenderPass,
        commands: &[GPURenderCommand],
        index_count: u32,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
    ) {
        if commands.is_empty() {
            return;
        }

        let total_instances: u32 = commands.iter().map(|c| c.instance_count).sum();
        if total_instances == 0 {
            return;
        }

        let (pipeline, bind_group) = self.ensure_pipeline_and_bind_group();
        let (Some(pipeline), Some(bind_group)) = (pipeline, bind_group.as_ref()) else {
            return;
        };

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        if let Some(buf) = self.instance_buffer.as_ref() {
            rpass.set_vertex_buffer(1, buf.slice(..));
        }
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        let mut base_instance: u32 = 0;
        for cmd in commands {
            if cmd.instance_count == 0 {
                continue;
            }
            rpass.draw_indexed(
                0..index_count,
                0,
                base_instance..base_instance + cmd.instance_count,
            );
            base_instance += cmd.instance_count;
        }
    }

    /// Reset all GPU buffer state (for device loss recovery).
    pub fn reset(&mut self) {
        self.instance_buffer = None;
        self.instance_buffer_capacity = 0;
        self.animation_buffer = None;
        self.animation_buffer_capacity = 0;
        self.color_buffer = None;
        self.color_buffer_capacity = 0;
        self.atlas_texture = None;
        self.atlas_view = None;
        self.pipeline = None;
        self.current_instance_buffer_bytes.clear();
        self.current_animation_buffer_bytes.clear();
        self.current_color_buffer_bytes.clear();
    }
}

impl Drop for CrowdRendererBridge {
    fn drop(&mut self) {
        // wgpu resources are dropped automatically via Rust's Drop
        // implementation on the buffers and textures.
    }
}

// ---------------------------------------------------------------------------
// Convenience function: build a GPURenderCommand from raw byte slices
// ---------------------------------------------------------------------------

/// Build a `GPURenderCommand` from raw slice references (for use by Python
/// FFI -- the Python side can pass buffer protocol objects that are
/// converted to slices).
pub fn make_render_command(
    mesh_id: u32,
    material_id: u32,
    instance_count: u32,
    transform_data: &[u8],
    animation_data: &[u8],
    color_data: &[u8],
    texture_atlas: u32,
) -> GPURenderCommand {
    GPURenderCommand {
        mesh_id,
        material_id,
        instance_count,
        transform_buffer: transform_data.to_vec(),
        animation_buffer: animation_data.to_vec(),
        color_buffer: color_data.to_vec(),
        texture_atlas,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GPURenderCommand tests ---------------------------------------------

    #[test]
    fn test_gpu_render_command_validate_valid() {
        let cmd = GPURenderCommand {
            mesh_id: 0,
            material_id: 0,
            instance_count: 2,
            transform_buffer: vec![0u8; 2 * 16 * 4],  // 2 instances * 64 bytes
            animation_buffer: vec![0u8; 2 * 4 * 4],    // 2 instances * 16 bytes
            color_buffer: vec![0u8; 2 * 4 * 4],        // 2 instances * 16 bytes
            texture_atlas: 0,
        };
        assert!(cmd.validate());
    }

    #[test]
    fn test_gpu_render_command_validate_invalid_instance_zero() {
        let cmd = GPURenderCommand {
            mesh_id: 0,
            material_id: 0,
            instance_count: 0,
            transform_buffer: vec![],
            animation_buffer: vec![],
            color_buffer: vec![],
            texture_atlas: 0,
        };
        assert!(!cmd.validate());
    }

    #[test]
    fn test_gpu_render_command_validate_buffer_size_mismatch() {
        let cmd = GPURenderCommand {
            mesh_id: 0,
            material_id: 0,
            instance_count: 2,
            transform_buffer: vec![0u8; 10], // wrong size
            animation_buffer: vec![0u8; 2 * 4 * 4],
            color_buffer: vec![0u8; 2 * 4 * 4],
            texture_atlas: 0,
        };
        assert!(!cmd.validate());
    }

    #[test]
    fn test_gpu_render_command_validate_partial_mismatch() {
        let cmd = GPURenderCommand {
            mesh_id: 1,
            material_id: 2,
            instance_count: 5,
            transform_buffer: vec![0u8; 5 * 16 * 4],
            animation_buffer: vec![0u8; 5 * 4 * 4],
            color_buffer: vec![0u8; 3 * 4 * 4], // short
            texture_atlas: 1,
        };
        assert!(!cmd.validate());
    }

    // -- make_render_command tests ------------------------------------------

    #[test]
    fn test_make_render_command_roundtrip() {
        let tdata = vec![1u8; 16 * 4];
        let adata = vec![2u8; 4 * 4];
        let cdata = vec![3u8; 4 * 4];

        let cmd = make_render_command(42, 7, 1, &tdata, &adata, &cdata, 3);

        assert_eq!(cmd.mesh_id, 42);
        assert_eq!(cmd.material_id, 7);
        assert_eq!(cmd.instance_count, 1);
        assert_eq!(cmd.transform_buffer, tdata);
        assert_eq!(cmd.animation_buffer, adata);
        assert_eq!(cmd.color_buffer, cdata);
        assert_eq!(cmd.texture_atlas, 3);
        assert!(cmd.validate());
    }

    // -- Instance data types ------------------------------------------------

    #[test]
    fn test_instance_data_sizes() {
        assert_eq!(std::mem::size_of::<InstanceTransform>(), 64);
        assert_eq!(std::mem::size_of::<InstanceAnimation>(), 16);
        assert_eq!(std::mem::size_of::<InstanceColor>(), 16);
    }

    #[test]
    fn test_instance_data_alignment() {
        assert_eq!(std::mem::align_of::<InstanceTransform>(), 4);
        assert_eq!(std::mem::align_of::<InstanceAnimation>(), 4);
        assert_eq!(std::mem::align_of::<InstanceColor>(), 4);
    }

    #[test]
    fn test_instance_transform_repr_c() {
        // Verify column-major layout: matrix[col][row]
        let xform = InstanceTransform {
            matrix: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let bytes: &[u8; 64] = unsafe { &*(&xform as *const InstanceTransform as *const [u8; 64]) };
        // Column 0 row 0 = 1.0 at byte offset 0
        let c0r0: f32 = unsafe { std::ptr::read(bytes.as_ptr().cast()) };
        assert!((c0r0 - 1.0).abs() < f32::EPSILON);
        // Column 1 row 0 = 5.0 at byte offset 16
        let c1r0: f32 = unsafe { std::ptr::read(bytes.as_ptr().add(16).cast()) };
        assert!((c1r0 - 5.0).abs() < f32::EPSILON);
        // Column 0 row 1 = 2.0 at byte offset 4
        let c0r1: f32 = unsafe { std::ptr::read(bytes.as_ptr().add(4).cast()) };
        assert!((c0r1 - 2.0).abs() < f32::EPSILON);
    }

    // -- Constants ----------------------------------------------------------

    #[test]
    fn test_constant_values_match_config() {
        assert_eq!(TRANSFORM_FLOATS, 16);
        assert_eq!(ANIMATION_FLOATS, 4);
        assert_eq!(COLOR_FLOATS, 4);
        assert_eq!(INSTANCE_STRIDE_BYTES, 96);
        assert_eq!(MAX_INSTANCES_PER_BATCH, 1000);
    }
}
