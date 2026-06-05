//! GPU Shaders for Crowd Animation (T-AN-8.3)
//!
//! This module provides GPU shaders for efficient crowd animation rendering,
//! including animation texture sampling, vertex skinning, and LOD variants.
//!
//! # Architecture
//!
//! ```text
//! Animation Texture Atlas          Instance Buffer          Uniform Buffer
//!        |                              |                        |
//!        v                              v                        v
//! +----------------+   +----------------------------+   +----------------+
//! | RGBA32F Tex2D  |   | CrowdInstance[N]           |   | time, camera   |
//! | bone_count * 2 |   | - position, rotation       |   | lod_distances  |
//! | x frame_count  |   | - animation_id, anim_time  |   +----------------+
//! +----------------+   | - lod_level, flags         |          |
//!        |             +----------------------------+          |
//!        |                              |                      |
//!        +----------+-------------------+-----+----------------+
//!                   |                         |
//!                   v                         v
//!        +--------------------+   +------------------------+
//!        | Animation Sampler  |   | LOD Shader Selection   |
//!        | - bilinear interp  |   | - LOD 0: full bones    |
//!        | - blend 2 anims    |   | - LOD 1: reduced bones |
//!        | - time offset      |   | - LOD 2: impostor      |
//!        +--------------------+   +------------------------+
//!                   |                         |
//!                   v                         v
//!        +--------------------+   +------------------------+
//!        | Vertex Skinning    |   | Output Vertex Buffer   |
//!        | - LBS per vertex   |   | - transformed pos      |
//!        | - bone weights     |   | - transformed normal   |
//!        +--------------------+   +------------------------+
//! ```
//!
//! # Shader Permutations
//!
//! The system supports compile-time feature flags:
//!
//! | Flag | Description |
//! |------|-------------|
//! | `BLEND_ANIMATIONS` | Enable 2-way animation blending |
//! | `USE_IMPOSTOR` | Use impostor/billboard rendering |
//! | `SMOOTH_LOD` | Enable smooth LOD transitions |
//! | `DEBUG_OUTPUT` | Output debug colors |
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::crowd_shaders::{
//!     CrowdAnimationShader, ShaderConfig, LodLevel,
//! };
//!
//! // Create shader manager
//! let config = ShaderConfig::default();
//! let shader = CrowdAnimationShader::new(64, config);
//!
//! // Generate shader source for LOD 0
//! let source = shader.generate_source(LodLevel::FullSkeleton);
//!
//! // Create bind group layout
//! let layout = shader.bind_group_layout_descriptor();
//! ```

use std::collections::HashMap;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum bones supported in full skeleton LOD.
pub const MAX_BONES_FULL: u32 = 256;

/// Maximum bones supported in simplified LOD.
pub const MAX_BONES_SIMPLIFIED: u32 = 32;

/// Maximum bone influences per vertex.
pub const MAX_BONE_INFLUENCES: u32 = 4;

/// Workgroup size for compute shaders.
pub const WORKGROUP_SIZE: u32 = 64;

/// Size of crowd uniforms in bytes.
pub const CROWD_UNIFORMS_SIZE: usize = 80;

/// Animation texture binding slot.
pub const BINDING_ANIMATION_TEXTURE: u32 = 0;

/// Instance buffer binding slot.
pub const BINDING_INSTANCE_BUFFER: u32 = 1;

/// Output vertex buffer binding slot.
pub const BINDING_OUTPUT_VERTICES: u32 = 2;

/// Uniform buffer binding slot.
pub const BINDING_UNIFORMS: u32 = 3;

/// Input vertex buffer binding slot (for skinning).
pub const BINDING_INPUT_VERTICES: u32 = 4;

/// Bone weights buffer binding slot.
pub const BINDING_BONE_WEIGHTS: u32 = 5;

/// Secondary animation texture binding slot (for blending).
pub const BINDING_ANIMATION_TEXTURE_B: u32 = 6;

// ---------------------------------------------------------------------------
// LodLevel
// ---------------------------------------------------------------------------

/// Level of detail for crowd animation rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LodLevel {
    /// Full skeleton with all bones (LOD 0).
    #[default]
    FullSkeleton = 0,

    /// Simplified skeleton with major joints only (LOD 1).
    Simplified = 1,

    /// Billboard/impostor rendering (LOD 2).
    Impostor = 2,
}

impl LodLevel {
    /// Get the bone count for this LOD level.
    pub fn bone_count(&self, max_bones: u32) -> u32 {
        match self {
            Self::FullSkeleton => max_bones,
            Self::Simplified => max_bones.min(MAX_BONES_SIMPLIFIED),
            Self::Impostor => 0,
        }
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::FullSkeleton => "full_skeleton",
            Self::Simplified => "simplified",
            Self::Impostor => "impostor",
        }
    }

    /// Parse from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::FullSkeleton),
            1 => Some(Self::Simplified),
            2 => Some(Self::Impostor),
            _ => None,
        }
    }

    /// Get all LOD levels.
    pub fn all() -> [LodLevel; 3] {
        [Self::FullSkeleton, Self::Simplified, Self::Impostor]
    }
}

impl fmt::Display for LodLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ShaderFeatures
// ---------------------------------------------------------------------------

/// Shader permutation feature flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShaderFeatures {
    /// Enable 2-way animation blending.
    pub blend_animations: bool,

    /// Use impostor/billboard rendering (overrides LOD).
    pub use_impostor: bool,

    /// Enable smooth LOD transitions.
    pub smooth_lod: bool,

    /// Output debug colors based on animation/LOD.
    pub debug_output: bool,

    /// Use half-precision for animation texture sampling.
    pub use_f16_textures: bool,

    /// Enable dual quaternion skinning (vs linear blend).
    pub dual_quaternion_skinning: bool,
}

impl ShaderFeatures {
    /// Create default features.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable animation blending.
    pub fn with_blend_animations(mut self, enable: bool) -> Self {
        self.blend_animations = enable;
        self
    }

    /// Enable impostor mode.
    pub fn with_impostor(mut self, enable: bool) -> Self {
        self.use_impostor = enable;
        self
    }

    /// Enable smooth LOD transitions.
    pub fn with_smooth_lod(mut self, enable: bool) -> Self {
        self.smooth_lod = enable;
        self
    }

    /// Enable debug output.
    pub fn with_debug_output(mut self, enable: bool) -> Self {
        self.debug_output = enable;
        self
    }

    /// Generate #define statements for these features.
    pub fn to_defines(&self) -> Vec<String> {
        let mut defines = Vec::new();

        if self.blend_animations {
            defines.push("#define BLEND_ANIMATIONS 1".to_string());
        }
        if self.use_impostor {
            defines.push("#define USE_IMPOSTOR 1".to_string());
        }
        if self.smooth_lod {
            defines.push("#define SMOOTH_LOD 1".to_string());
        }
        if self.debug_output {
            defines.push("#define DEBUG_OUTPUT 1".to_string());
        }
        if self.use_f16_textures {
            defines.push("#define USE_F16_TEXTURES 1".to_string());
        }
        if self.dual_quaternion_skinning {
            defines.push("#define DUAL_QUATERNION_SKINNING 1".to_string());
        }

        defines
    }

    /// Generate a unique key for shader caching.
    pub fn cache_key(&self) -> u64 {
        let mut key = 0u64;
        if self.blend_animations { key |= 1 << 0; }
        if self.use_impostor { key |= 1 << 1; }
        if self.smooth_lod { key |= 1 << 2; }
        if self.debug_output { key |= 1 << 3; }
        if self.use_f16_textures { key |= 1 << 4; }
        if self.dual_quaternion_skinning { key |= 1 << 5; }
        key
    }

    /// Parse features from cache key.
    pub fn from_cache_key(key: u64) -> Self {
        Self {
            blend_animations: (key & (1 << 0)) != 0,
            use_impostor: (key & (1 << 1)) != 0,
            smooth_lod: (key & (1 << 2)) != 0,
            debug_output: (key & (1 << 3)) != 0,
            use_f16_textures: (key & (1 << 4)) != 0,
            dual_quaternion_skinning: (key & (1 << 5)) != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderConfig
// ---------------------------------------------------------------------------

/// Configuration for crowd animation shaders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderConfig {
    /// Maximum number of bones.
    pub max_bones: u32,

    /// Sample rate of animation textures.
    pub sample_rate: f32,

    /// Shader features.
    pub features: ShaderFeatures,

    /// Workgroup size for compute shaders.
    pub workgroup_size: u32,

    /// Maximum instances per dispatch.
    pub max_instances: u32,
}

impl ShaderConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self {
            max_bones: 64,
            sample_rate: 30.0,
            features: ShaderFeatures::default(),
            workgroup_size: WORKGROUP_SIZE,
            max_instances: 10_000,
        }
    }

    /// Set maximum bone count.
    pub fn with_max_bones(mut self, bones: u32) -> Self {
        self.max_bones = bones.min(MAX_BONES_FULL);
        self
    }

    /// Set sample rate.
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate.max(1.0);
        self
    }

    /// Set shader features.
    pub fn with_features(mut self, features: ShaderFeatures) -> Self {
        self.features = features;
        self
    }

    /// Set workgroup size.
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size.max(1).min(256);
        self
    }

    /// Set maximum instances.
    pub fn with_max_instances(mut self, instances: u32) -> Self {
        self.max_instances = instances;
        self
    }

    /// Generate unique cache key for this configuration.
    pub fn cache_key(&self) -> u64 {
        let mut key = self.features.cache_key();
        key ^= (self.max_bones as u64) << 8;
        key ^= (self.workgroup_size as u64) << 16;
        key
    }
}

impl Default for ShaderConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CrowdUniforms
// ---------------------------------------------------------------------------

/// Uniform buffer data for crowd animation shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CrowdUniforms {
    /// Global animation time (seconds).
    pub time: f32,
    /// Delta time since last frame.
    pub delta_time: f32,
    /// Number of bones in skeleton.
    pub bone_count: u32,
    /// Frame count in animation texture.
    pub frame_count: u32,

    /// Camera position (world space).
    pub camera_pos: [f32; 3],
    /// Animation sample rate.
    pub sample_rate: f32,

    /// LOD distance thresholds.
    pub lod_distances: [f32; 3],
    /// Blend weight (0-1) for animation blending.
    pub blend_weight: f32,

    /// Animation texture dimensions.
    pub texture_width: u32,
    pub texture_height: u32,
    /// LOD transition range.
    pub lod_transition_range: f32,
    /// Flags.
    pub flags: u32,
}

impl CrowdUniforms {
    /// Create new uniforms with default values.
    pub fn new() -> Self {
        Self {
            time: 0.0,
            delta_time: 0.0,
            bone_count: 64,
            frame_count: 30,
            camera_pos: [0.0, 0.0, 0.0],
            sample_rate: 30.0,
            lod_distances: [20.0, 50.0, 100.0],
            blend_weight: 0.0,
            texture_width: 256,
            texture_height: 128,
            lod_transition_range: 2.0,
            flags: 0,
        }
    }

    /// Set time values.
    pub fn with_time(mut self, time: f32, delta_time: f32) -> Self {
        self.time = time;
        self.delta_time = delta_time;
        self
    }

    /// Set camera position.
    pub fn with_camera_pos(mut self, pos: [f32; 3]) -> Self {
        self.camera_pos = pos;
        self
    }

    /// Set animation parameters.
    pub fn with_animation(mut self, bone_count: u32, frame_count: u32, sample_rate: f32) -> Self {
        self.bone_count = bone_count;
        self.frame_count = frame_count;
        self.sample_rate = sample_rate;
        self
    }

    /// Set LOD distances.
    pub fn with_lod_distances(mut self, distances: [f32; 3]) -> Self {
        self.lod_distances = distances;
        self
    }

    /// Set blend weight.
    pub fn with_blend_weight(mut self, weight: f32) -> Self {
        self.blend_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set texture dimensions.
    pub fn with_texture_dimensions(mut self, width: u32, height: u32) -> Self {
        self.texture_width = width;
        self.texture_height = height;
        self
    }

    /// Convert to bytes for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ---------------------------------------------------------------------------
// BindGroupLayoutEntry helpers
// ---------------------------------------------------------------------------

/// Descriptor for a bind group layout entry.
#[derive(Clone, Debug, PartialEq)]
pub struct BindGroupLayoutEntryDescriptor {
    /// Binding index.
    pub binding: u32,
    /// Visibility (vertex, fragment, compute).
    pub visibility: ShaderStage,
    /// Binding type.
    pub ty: BindingType,
}

/// Shader stage visibility flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderStage(pub u32);

impl ShaderStage {
    pub const NONE: Self = Self(0);
    pub const VERTEX: Self = Self(1);
    pub const FRAGMENT: Self = Self(2);
    pub const COMPUTE: Self = Self(4);
    pub const ALL: Self = Self(7);

    /// Check if this includes vertex stage.
    pub fn has_vertex(self) -> bool { (self.0 & 1) != 0 }
    /// Check if this includes fragment stage.
    pub fn has_fragment(self) -> bool { (self.0 & 2) != 0 }
    /// Check if this includes compute stage.
    pub fn has_compute(self) -> bool { (self.0 & 4) != 0 }
}

impl std::ops::BitOr for ShaderStage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Binding type descriptor.
#[derive(Clone, Debug, PartialEq)]
pub enum BindingType {
    /// Uniform buffer.
    UniformBuffer { size: u64 },
    /// Storage buffer (read-only).
    StorageBufferReadOnly { size: u64 },
    /// Storage buffer (read-write).
    StorageBuffer { size: u64 },
    /// Texture (2D, filtered).
    Texture2D { filterable: bool },
    /// Texture (2D array).
    Texture2DArray { filterable: bool },
    /// Sampler.
    Sampler { filtering: bool },
}

// ---------------------------------------------------------------------------
// ShaderPermutation
// ---------------------------------------------------------------------------

/// A specific shader permutation with LOD and features.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderPermutation {
    /// LOD level.
    pub lod: LodLevel,
    /// Feature flags.
    pub features: ShaderFeatures,
}

impl ShaderPermutation {
    /// Create a new permutation.
    pub fn new(lod: LodLevel, features: ShaderFeatures) -> Self {
        Self { lod, features }
    }

    /// Generate a unique cache key.
    pub fn cache_key(&self) -> u64 {
        let lod_bits = (self.lod as u64) << 56;
        lod_bits | self.features.cache_key()
    }
}

// ---------------------------------------------------------------------------
// ShaderCache
// ---------------------------------------------------------------------------

/// Cache for compiled shader variants.
#[derive(Debug, Default)]
pub struct ShaderCache {
    /// Cached shader sources by permutation key.
    sources: HashMap<u64, String>,
    /// Hit count.
    hits: u64,
    /// Miss count.
    misses: u64,
}

impl ShaderCache {
    /// Create a new shader cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or generate shader source for a permutation.
    pub fn get_or_insert<F>(&mut self, perm: &ShaderPermutation, generator: F) -> &str
    where
        F: FnOnce() -> String,
    {
        let key = perm.cache_key();

        if self.sources.contains_key(&key) {
            self.hits += 1;
            return self.sources.get(&key).unwrap();
        }

        self.misses += 1;
        let source = generator();
        self.sources.insert(key, source);
        self.sources.get(&key).unwrap()
    }

    /// Check if a permutation is cached.
    pub fn contains(&self, perm: &ShaderPermutation) -> bool {
        self.sources.contains_key(&perm.cache_key())
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get number of cached shaders.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WGSL Shader Source Generation
// ---------------------------------------------------------------------------

/// Animation texture sampling WGSL code.
pub const WGSL_ANIMATION_SAMPLING: &str = r#"
// Animation Texture Sampling Functions
// =====================================

// Sample bone transform from animation texture
fn sample_bone_transform(
    bone: u32,
    frame: f32,
    anim_tex: texture_2d<f32>,
    tex_width: u32,
    tex_height: u32,
) -> mat4x4<f32> {
    // Calculate frame indices for interpolation
    let frame_clamped = clamp(frame, 0.0, f32(tex_width - 1u));
    let frame0 = u32(floor(frame_clamped));
    let frame1 = min(frame0 + 1u, tex_width - 1u);
    let t = fract(frame_clamped);

    // Row indices for position and rotation
    let pos_row = bone * 2u;
    let rot_row = bone * 2u + 1u;

    // Sample position at both frames
    let pos0 = textureLoad(anim_tex, vec2<i32>(i32(frame0), i32(pos_row)), 0);
    let pos1 = textureLoad(anim_tex, vec2<i32>(i32(frame1), i32(pos_row)), 0);
    let position = mix(pos0, pos1, t);

    // Sample rotation (quaternion) at both frames
    let rot0 = textureLoad(anim_tex, vec2<i32>(i32(frame0), i32(rot_row)), 0);
    let rot1 = textureLoad(anim_tex, vec2<i32>(i32(frame1), i32(rot_row)), 0);
    let rotation = nlerp_quat(rot0, rot1, t);

    // Compose transform matrix
    return compose_transform(position.xyz, rotation, position.w);
}

// Normalized linear interpolation for quaternions
fn nlerp_quat(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    // Handle antipodal quaternions (shortest path)
    var b_adj = b;
    if (dot(a, b) < 0.0) {
        b_adj = -b;
    }

    // Linear interpolation
    let result = mix(a, b_adj, t);

    // Normalize
    return normalize(result);
}

// Compose transform matrix from position, quaternion rotation, and uniform scale
fn compose_transform(position: vec3<f32>, rotation: vec4<f32>, scale: f32) -> mat4x4<f32> {
    let qx = rotation.x;
    let qy = rotation.y;
    let qz = rotation.z;
    let qw = rotation.w;

    let s = scale;

    // Rotation matrix from quaternion, scaled
    let m00 = s * (1.0 - 2.0 * (qy * qy + qz * qz));
    let m01 = s * (2.0 * (qx * qy - qz * qw));
    let m02 = s * (2.0 * (qx * qz + qy * qw));

    let m10 = s * (2.0 * (qx * qy + qz * qw));
    let m11 = s * (1.0 - 2.0 * (qx * qx + qz * qz));
    let m12 = s * (2.0 * (qy * qz - qx * qw));

    let m20 = s * (2.0 * (qx * qz - qy * qw));
    let m21 = s * (2.0 * (qy * qz + qx * qw));
    let m22 = s * (1.0 - 2.0 * (qx * qx + qy * qy));

    return mat4x4<f32>(
        vec4<f32>(m00, m10, m20, 0.0),
        vec4<f32>(m01, m11, m21, 0.0),
        vec4<f32>(m02, m12, m22, 0.0),
        vec4<f32>(position.x, position.y, position.z, 1.0)
    );
}
"#;

/// Animation blending WGSL code (optional).
pub const WGSL_ANIMATION_BLENDING: &str = r#"
#ifdef BLEND_ANIMATIONS
// Blend two animation samples
fn sample_blended_bone_transform(
    bone: u32,
    frame_a: f32,
    frame_b: f32,
    blend_weight: f32,
    anim_tex_a: texture_2d<f32>,
    anim_tex_b: texture_2d<f32>,
    tex_width: u32,
    tex_height: u32,
) -> mat4x4<f32> {
    let transform_a = sample_bone_transform(bone, frame_a, anim_tex_a, tex_width, tex_height);
    let transform_b = sample_bone_transform(bone, frame_b, anim_tex_b, tex_width, tex_height);

    // Extract position
    let pos_a = transform_a[3].xyz;
    let pos_b = transform_b[3].xyz;
    let blended_pos = mix(pos_a, pos_b, blend_weight);

    // For rotation, extract quaternion and blend
    let rot_a = mat_to_quat(transform_a);
    let rot_b = mat_to_quat(transform_b);
    let blended_rot = nlerp_quat(rot_a, rot_b, blend_weight);

    // Scale (from diagonal)
    let scale_a = length(transform_a[0].xyz);
    let scale_b = length(transform_b[0].xyz);
    let blended_scale = mix(scale_a, scale_b, blend_weight);

    return compose_transform(blended_pos, blended_rot, blended_scale);
}

// Extract quaternion from rotation matrix
fn mat_to_quat(m: mat4x4<f32>) -> vec4<f32> {
    let trace = m[0][0] + m[1][1] + m[2][2];

    var qw: f32;
    var qx: f32;
    var qy: f32;
    var qz: f32;

    if (trace > 0.0) {
        let s = 0.5 / sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (m[2][1] - m[1][2]) * s;
        qy = (m[0][2] - m[2][0]) * s;
        qz = (m[1][0] - m[0][1]) * s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        let s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        qw = (m[2][1] - m[1][2]) / s;
        qx = 0.25 * s;
        qy = (m[0][1] + m[1][0]) / s;
        qz = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        let s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        qw = (m[0][2] - m[2][0]) / s;
        qx = (m[0][1] + m[1][0]) / s;
        qy = 0.25 * s;
        qz = (m[1][2] + m[2][1]) / s;
    } else {
        let s = 2.0 * sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        qw = (m[1][0] - m[0][1]) / s;
        qx = (m[0][2] + m[2][0]) / s;
        qy = (m[1][2] + m[2][1]) / s;
        qz = 0.25 * s;
    }

    return normalize(vec4<f32>(qx, qy, qz, qw));
}
#endif
"#;

/// Vertex skinning compute shader WGSL code.
pub const WGSL_VERTEX_SKINNING: &str = r#"
// Vertex Skinning Compute Shader
// ===============================

struct CrowdInstance {
    position: vec3<f32>,
    scale: f32,
    rotation: vec4<f32>,
    animation_id: u32,  // Packed: (anim_id << 16) | lod_level
    animation_time: f32,
    _padding: vec2<f32>,
}

struct InputVertex {
    position: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
    bone_indices: vec4<u32>,
    bone_weights: vec4<f32>,
}

struct OutputVertex {
    position: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
}

struct Uniforms {
    time: f32,
    delta_time: f32,
    bone_count: u32,
    frame_count: u32,
    camera_pos: vec3<f32>,
    sample_rate: f32,
    lod_distances: vec3<f32>,
    blend_weight: f32,
    texture_width: u32,
    texture_height: u32,
    lod_transition_range: f32,
    flags: u32,
}

@group(0) @binding(0) var anim_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> instances: array<CrowdInstance>;
@group(0) @binding(2) var<storage, read_write> output_vertices: array<OutputVertex>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> input_vertices: array<InputVertex>;

@compute @workgroup_size(64)
fn cs_skin_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_index = gid.x;
    let instance_index = gid.y;

    // Bounds check
    let vertex_count = arrayLength(&input_vertices);
    let instance_count = arrayLength(&instances);

    if (vertex_index >= vertex_count || instance_index >= instance_count) {
        return;
    }

    let instance = instances[instance_index];
    let vertex = input_vertices[vertex_index];

    // Calculate animation frame
    let frame = instance.animation_time * uniforms.sample_rate;

    // Extract LOD level from packed animation_id
    let lod_level = instance.animation_id & 0xFFu;

    // Skip skinning for impostors
    if (lod_level == 2u) {
        let output_index = instance_index * vertex_count + vertex_index;
        output_vertices[output_index].position = vertex.position;
        output_vertices[output_index].normal = vertex.normal;
        return;
    }

    // Accumulate skinned position and normal
    var skinned_pos = vec3<f32>(0.0);
    var skinned_normal = vec3<f32>(0.0);

    // Linear blend skinning with 4 bone influences
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bone_index = vertex.bone_indices[i];
        let weight = vertex.bone_weights[i];

        if (weight > 0.0001) {
            let bone_transform = sample_bone_transform(
                bone_index,
                frame,
                anim_texture,
                uniforms.texture_width,
                uniforms.texture_height
            );

            // Transform position
            let transformed_pos = (bone_transform * vec4<f32>(vertex.position, 1.0)).xyz;
            skinned_pos = skinned_pos + transformed_pos * weight;

            // Transform normal (rotation only)
            let rotation_only = mat3x3<f32>(
                bone_transform[0].xyz,
                bone_transform[1].xyz,
                bone_transform[2].xyz
            );
            skinned_normal = skinned_normal + (rotation_only * vertex.normal) * weight;
        }
    }

    // Apply instance transform
    let instance_transform = compose_transform(
        instance.position,
        instance.rotation,
        instance.scale
    );

    let final_pos = (instance_transform * vec4<f32>(skinned_pos, 1.0)).xyz;
    let instance_rot = mat3x3<f32>(
        instance_transform[0].xyz,
        instance_transform[1].xyz,
        instance_transform[2].xyz
    );
    let final_normal = normalize(instance_rot * skinned_normal);

    // Write output
    let output_index = instance_index * vertex_count + vertex_index;
    output_vertices[output_index].position = final_pos;
    output_vertices[output_index].normal = final_normal;
}
"#;

/// Impostor rendering WGSL code (LOD 2).
pub const WGSL_IMPOSTOR: &str = r#"
// Impostor/Billboard Rendering
// =============================

struct ImpostorVertex {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) instance_index: u32,
}

struct ImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) instance_index: u32,
}

@vertex
fn vs_impostor(input: ImpostorVertex) -> ImpostorOutput {
    var output: ImpostorOutput;

    // Billboard quad vertices (2 triangles)
    let quad_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, 0.0),
        vec2<f32>(0.5, 0.0),
        vec2<f32>(-0.5, 2.0),
        vec2<f32>(0.5, 2.0)
    );

    let instance = instances[input.instance_index];

    // Camera-facing billboard
    let to_camera = normalize(uniforms.camera_pos - instance.position);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_camera));
    let billboard_up = cross(to_camera, right);

    // Calculate billboard position
    let vertex_id = u32(input.position.x); // Encoded in position.x
    let offset = quad_offsets[vertex_id];
    let world_pos = instance.position
        + right * offset.x * instance.scale
        + billboard_up * offset.y * instance.scale;

    // Project to clip space (simplified - would use actual view/proj matrices)
    output.clip_position = vec4<f32>(world_pos, 1.0);
    output.uv = vec2<f32>(f32(vertex_id & 1u), f32((vertex_id >> 1u) & 1u));
    output.instance_index = input.instance_index;

    return output;
}

@fragment
fn fs_impostor(input: ImpostorOutput) -> @location(0) vec4<f32> {
    // Sample impostor texture based on animation frame
    // (Simplified - would sample from pre-rendered impostor atlas)
    return vec4<f32>(1.0, 0.8, 0.6, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// CrowdAnimationShader
// ---------------------------------------------------------------------------

/// Main shader manager for crowd animation rendering.
#[derive(Debug)]
pub struct CrowdAnimationShader {
    /// Configuration.
    pub config: ShaderConfig,

    /// Shader cache for compiled variants.
    cache: ShaderCache,
}

impl CrowdAnimationShader {
    /// Create a new shader manager.
    pub fn new(max_bones: u32, config: ShaderConfig) -> Self {
        Self {
            config: config.with_max_bones(max_bones),
            cache: ShaderCache::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(max_bones: u32) -> Self {
        Self::new(max_bones, ShaderConfig::default())
    }

    /// Get configuration.
    pub fn config(&self) -> &ShaderConfig {
        &self.config
    }

    /// Generate shader source for a specific LOD level.
    pub fn generate_source(&self, lod: LodLevel) -> String {
        self.generate_source_with_features(lod, &self.config.features)
    }

    /// Generate shader source with specific features.
    pub fn generate_source_with_features(&self, lod: LodLevel, features: &ShaderFeatures) -> String {
        let mut source = String::with_capacity(8192);

        // Header comment
        source.push_str("// TRINITY Crowd Animation Shader\n");
        source.push_str(&format!("// LOD: {}\n", lod));
        source.push_str(&format!("// Generated with {} bones\n", self.config.max_bones));
        source.push_str("\n");

        // Feature defines
        let defines = features.to_defines();
        for define in &defines {
            source.push_str(define);
            source.push('\n');
        }

        // LOD-specific defines
        source.push_str(&format!("#define LOD_LEVEL {}\n", lod as u32));
        source.push_str(&format!("#define MAX_BONES {}u\n", self.config.max_bones));
        source.push_str(&format!("#define WORKGROUP_SIZE {}u\n", self.config.workgroup_size));
        source.push('\n');

        // Common animation sampling code
        source.push_str(WGSL_ANIMATION_SAMPLING);
        source.push('\n');

        // Animation blending (conditional)
        if features.blend_animations {
            // Remove preprocessor directives for WGSL compatibility
            let blending_code = WGSL_ANIMATION_BLENDING
                .replace("#ifdef BLEND_ANIMATIONS", "")
                .replace("#endif", "");
            source.push_str(&blending_code);
            source.push('\n');
        }

        // LOD-specific code
        match lod {
            LodLevel::FullSkeleton | LodLevel::Simplified => {
                source.push_str(WGSL_VERTEX_SKINNING);
            }
            LodLevel::Impostor => {
                source.push_str(WGSL_IMPOSTOR);
            }
        }

        source
    }

    /// Generate shader for a specific permutation.
    pub fn generate_permutation(&mut self, perm: &ShaderPermutation) -> &str {
        let lod = perm.lod;
        let features = perm.features.clone();
        let config = &self.config;

        self.cache.get_or_insert(perm, || {
            let shader = CrowdAnimationShader::new(config.max_bones, config.clone());
            shader.generate_source_with_features(lod, &features)
        })
    }

    /// Check if a permutation is cached.
    pub fn is_cached(&self, perm: &ShaderPermutation) -> bool {
        self.cache.contains(perm)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache.stats()
    }

    /// Clear shader cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get bind group layout entries for the skinning compute shader.
    pub fn bind_group_layout_entries(&self) -> Vec<BindGroupLayoutEntryDescriptor> {
        vec![
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_ANIMATION_TEXTURE,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::Texture2D { filterable: false },
            },
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_INSTANCE_BUFFER,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::StorageBufferReadOnly { size: 0 },
            },
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_OUTPUT_VERTICES,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::StorageBuffer { size: 0 },
            },
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_UNIFORMS,
                visibility: ShaderStage::COMPUTE | ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                ty: BindingType::UniformBuffer {
                    size: CROWD_UNIFORMS_SIZE as u64,
                },
            },
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_INPUT_VERTICES,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::StorageBufferReadOnly { size: 0 },
            },
            BindGroupLayoutEntryDescriptor {
                binding: BINDING_BONE_WEIGHTS,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::StorageBufferReadOnly { size: 0 },
            },
        ]
    }

    /// Get bind group layout entries for blending (adds secondary animation texture).
    pub fn bind_group_layout_entries_blend(&self) -> Vec<BindGroupLayoutEntryDescriptor> {
        let mut entries = self.bind_group_layout_entries();
        entries.push(BindGroupLayoutEntryDescriptor {
            binding: BINDING_ANIMATION_TEXTURE_B,
            visibility: ShaderStage::COMPUTE,
            ty: BindingType::Texture2D { filterable: false },
        });
        entries
    }

    /// Get the number of workgroups needed for a given vertex count.
    pub fn workgroups_for_vertices(&self, vertex_count: u32) -> u32 {
        (vertex_count + self.config.workgroup_size - 1) / self.config.workgroup_size
    }

    /// Get the number of workgroups for instances.
    pub fn workgroups_for_instances(&self, instance_count: u32) -> u32 {
        instance_count
    }

    /// Validate shader source using basic checks.
    pub fn validate_source(&self, source: &str) -> Result<(), ShaderValidationError> {
        // Check for required entry points
        if !source.contains("@compute") && !source.contains("@vertex") {
            return Err(ShaderValidationError::MissingEntryPoint);
        }

        // Check for unbalanced braces
        let open_braces = source.matches('{').count();
        let close_braces = source.matches('}').count();
        if open_braces != close_braces {
            return Err(ShaderValidationError::UnbalancedBraces {
                open: open_braces,
                close: close_braces,
            });
        }

        // Check for unbalanced parentheses
        let open_parens = source.matches('(').count();
        let close_parens = source.matches(')').count();
        if open_parens != close_parens {
            return Err(ShaderValidationError::UnbalancedParentheses {
                open: open_parens,
                close: close_parens,
            });
        }

        Ok(())
    }

    /// Get bone count for a specific LOD.
    pub fn bone_count_for_lod(&self, lod: LodLevel) -> u32 {
        lod.bone_count(self.config.max_bones)
    }
}

impl Default for CrowdAnimationShader {
    fn default() -> Self {
        Self::with_defaults(64)
    }
}

// ---------------------------------------------------------------------------
// ShaderValidationError
// ---------------------------------------------------------------------------

/// Errors during shader validation.
#[derive(Clone, Debug, PartialEq)]
pub enum ShaderValidationError {
    /// Missing entry point.
    MissingEntryPoint,
    /// Unbalanced braces.
    UnbalancedBraces { open: usize, close: usize },
    /// Unbalanced parentheses.
    UnbalancedParentheses { open: usize, close: usize },
    /// Invalid syntax.
    InvalidSyntax { message: String },
    /// Naga compilation error.
    NagaError { message: String },
}

impl fmt::Display for ShaderValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEntryPoint => write!(f, "shader has no entry point (@vertex, @fragment, or @compute)"),
            Self::UnbalancedBraces { open, close } => {
                write!(f, "unbalanced braces: {} open, {} close", open, close)
            }
            Self::UnbalancedParentheses { open, close } => {
                write!(f, "unbalanced parentheses: {} open, {} close", open, close)
            }
            Self::InvalidSyntax { message } => write!(f, "invalid syntax: {}", message),
            Self::NagaError { message } => write!(f, "naga compilation error: {}", message),
        }
    }
}

impl std::error::Error for ShaderValidationError {}

// ---------------------------------------------------------------------------
// ShaderCompiler utilities
// ---------------------------------------------------------------------------

/// Utility functions for shader compilation.
pub struct ShaderCompiler;

impl ShaderCompiler {
    /// Preprocess shader source with defines.
    pub fn preprocess(source: &str, defines: &[(&str, &str)]) -> String {
        let mut result = String::with_capacity(source.len() + 256);

        // Add defines
        for (name, value) in defines {
            result.push_str(&format!("#define {} {}\n", name, value));
        }

        result.push_str(source);
        result
    }

    /// Remove preprocessor directives (for pure WGSL).
    pub fn strip_preprocessor(source: &str) -> String {
        source
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate shader source with custom bone count.
    pub fn generate_skinning_shader(bone_count: u32, features: &ShaderFeatures) -> String {
        let shader = CrowdAnimationShader::new(bone_count, ShaderConfig::default().with_features(*features));
        shader.generate_source(LodLevel::FullSkeleton)
    }

    /// Count lines in shader source.
    pub fn line_count(source: &str) -> usize {
        source.lines().count()
    }

    /// Estimate shader complexity (rough heuristic).
    pub fn estimate_complexity(source: &str) -> u32 {
        let mut complexity = 0u32;

        // Count functions
        complexity += source.matches("fn ").count() as u32 * 10;

        // Count loops
        complexity += source.matches("for ").count() as u32 * 20;
        complexity += source.matches("while ").count() as u32 * 20;
        complexity += source.matches("loop ").count() as u32 * 15;

        // Count conditionals
        complexity += source.matches("if ").count() as u32 * 5;
        complexity += source.matches("else ").count() as u32 * 3;

        // Count texture operations
        complexity += source.matches("textureLoad").count() as u32 * 8;
        complexity += source.matches("textureSample").count() as u32 * 10;

        // Count matrix operations
        complexity += source.matches("mat4x4").count() as u32 * 5;
        complexity += source.matches("mat3x3").count() as u32 * 3;

        complexity
    }
}

// ---------------------------------------------------------------------------
// Bilinear interpolation helpers (CPU reference)
// ---------------------------------------------------------------------------

/// Bilinear interpolation helper for animation sampling (CPU reference).
pub fn bilinear_sample(
    data: &[[f32; 4]],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
) -> [f32; 4] {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);

    let x0 = x.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y0 = y.floor() as u32;
    let y1 = (y0 + 1).min(height - 1);

    let tx = x.fract();
    let ty = y.fract();

    let idx00 = (y0 * width + x0) as usize;
    let idx01 = (y0 * width + x1) as usize;
    let idx10 = (y1 * width + x0) as usize;
    let idx11 = (y1 * width + x1) as usize;

    // Bounds check
    let max_idx = data.len().saturating_sub(1);
    let v00 = data.get(idx00.min(max_idx)).copied().unwrap_or([0.0; 4]);
    let v01 = data.get(idx01.min(max_idx)).copied().unwrap_or([0.0; 4]);
    let v10 = data.get(idx10.min(max_idx)).copied().unwrap_or([0.0; 4]);
    let v11 = data.get(idx11.min(max_idx)).copied().unwrap_or([0.0; 4]);

    let mut result = [0.0f32; 4];
    for i in 0..4 {
        let lerp_x0 = v00[i] + (v01[i] - v00[i]) * tx;
        let lerp_x1 = v10[i] + (v11[i] - v10[i]) * tx;
        result[i] = lerp_x0 + (lerp_x1 - lerp_x0) * ty;
    }

    result
}

/// Normalized linear interpolation for quaternions (CPU reference).
pub fn nlerp_quat(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    // Handle antipodal quaternions
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let b_adj = if dot < 0.0 {
        [-b[0], -b[1], -b[2], -b[3]]
    } else {
        b
    };

    // Linear interpolation
    let mut result = [
        a[0] + (b_adj[0] - a[0]) * t,
        a[1] + (b_adj[1] - a[1]) * t,
        a[2] + (b_adj[2] - a[2]) * t,
        a[3] + (b_adj[3] - a[3]) * t,
    ];

    // Normalize
    let len = (result[0] * result[0]
        + result[1] * result[1]
        + result[2] * result[2]
        + result[3] * result[3])
        .sqrt();

    if len > 0.0001 {
        for v in &mut result {
            *v /= len;
        }
    } else {
        result = [0.0, 0.0, 0.0, 1.0];
    }

    result
}

/// Compose a 4x4 transform matrix from position, quaternion rotation, and scale.
pub fn compose_transform_matrix(
    position: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
) -> [[f32; 4]; 4] {
    let [qx, qy, qz, qw] = rotation;
    let s = scale;

    let m00 = s * (1.0 - 2.0 * (qy * qy + qz * qz));
    let m01 = s * (2.0 * (qx * qy - qz * qw));
    let m02 = s * (2.0 * (qx * qz + qy * qw));

    let m10 = s * (2.0 * (qx * qy + qz * qw));
    let m11 = s * (1.0 - 2.0 * (qx * qx + qz * qz));
    let m12 = s * (2.0 * (qy * qz - qx * qw));

    let m20 = s * (2.0 * (qx * qz - qy * qw));
    let m21 = s * (2.0 * (qy * qz + qx * qw));
    let m22 = s * (1.0 - 2.0 * (qx * qx + qy * qy));

    [
        [m00, m10, m20, 0.0],
        [m01, m11, m21, 0.0],
        [m02, m12, m22, 0.0],
        [position[0], position[1], position[2], 1.0],
    ]
}

/// Apply linear blend skinning to a vertex (CPU reference).
pub fn apply_lbs(
    position: [f32; 3],
    bone_indices: [u32; 4],
    bone_weights: [f32; 4],
    bone_matrices: &[[[f32; 4]; 4]],
) -> [f32; 3] {
    let mut result = [0.0f32; 3];

    for i in 0..4 {
        let weight = bone_weights[i];
        if weight > 0.0001 {
            let bone_idx = bone_indices[i] as usize;
            if bone_idx < bone_matrices.len() {
                let m = &bone_matrices[bone_idx];

                // Transform position by bone matrix
                let x = m[0][0] * position[0] + m[1][0] * position[1] + m[2][0] * position[2] + m[3][0];
                let y = m[0][1] * position[0] + m[1][1] * position[1] + m[2][1] * position[2] + m[3][1];
                let z = m[0][2] * position[0] + m[1][2] * position[1] + m[2][2] * position[2] + m[3][2];

                result[0] += x * weight;
                result[1] += y * weight;
                result[2] += z * weight;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ========================================================================
    // LodLevel Tests
    // ========================================================================

    #[test]
    fn test_lod_level_from_u8() {
        assert_eq!(LodLevel::from_u8(0), Some(LodLevel::FullSkeleton));
        assert_eq!(LodLevel::from_u8(1), Some(LodLevel::Simplified));
        assert_eq!(LodLevel::from_u8(2), Some(LodLevel::Impostor));
        assert_eq!(LodLevel::from_u8(3), None);
        assert_eq!(LodLevel::from_u8(255), None);
    }

    #[test]
    fn test_lod_level_bone_count() {
        assert_eq!(LodLevel::FullSkeleton.bone_count(64), 64);
        assert_eq!(LodLevel::FullSkeleton.bone_count(256), 256);
        assert_eq!(LodLevel::Simplified.bone_count(64), 32);
        assert_eq!(LodLevel::Simplified.bone_count(20), 20);
        assert_eq!(LodLevel::Impostor.bone_count(64), 0);
    }

    #[test]
    fn test_lod_level_name() {
        assert_eq!(LodLevel::FullSkeleton.name(), "full_skeleton");
        assert_eq!(LodLevel::Simplified.name(), "simplified");
        assert_eq!(LodLevel::Impostor.name(), "impostor");
    }

    #[test]
    fn test_lod_level_all() {
        let levels = LodLevel::all();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], LodLevel::FullSkeleton);
        assert_eq!(levels[1], LodLevel::Simplified);
        assert_eq!(levels[2], LodLevel::Impostor);
    }

    #[test]
    fn test_lod_level_display() {
        assert_eq!(format!("{}", LodLevel::FullSkeleton), "full_skeleton");
        assert_eq!(format!("{}", LodLevel::Impostor), "impostor");
    }

    // ========================================================================
    // ShaderFeatures Tests
    // ========================================================================

    #[test]
    fn test_shader_features_default() {
        let features = ShaderFeatures::default();
        assert!(!features.blend_animations);
        assert!(!features.use_impostor);
        assert!(!features.smooth_lod);
        assert!(!features.debug_output);
    }

    #[test]
    fn test_shader_features_builder() {
        let features = ShaderFeatures::new()
            .with_blend_animations(true)
            .with_impostor(true)
            .with_smooth_lod(true)
            .with_debug_output(true);

        assert!(features.blend_animations);
        assert!(features.use_impostor);
        assert!(features.smooth_lod);
        assert!(features.debug_output);
    }

    #[test]
    fn test_shader_features_to_defines() {
        let features = ShaderFeatures::new()
            .with_blend_animations(true)
            .with_debug_output(true);

        let defines = features.to_defines();
        assert!(defines.contains(&"#define BLEND_ANIMATIONS 1".to_string()));
        assert!(defines.contains(&"#define DEBUG_OUTPUT 1".to_string()));
        assert!(!defines.iter().any(|d| d.contains("SMOOTH_LOD")));
    }

    #[test]
    fn test_shader_features_cache_key() {
        let f1 = ShaderFeatures::default();
        let f2 = ShaderFeatures::new().with_blend_animations(true);
        let f3 = ShaderFeatures::new().with_blend_animations(true);

        assert_ne!(f1.cache_key(), f2.cache_key());
        assert_eq!(f2.cache_key(), f3.cache_key());
    }

    #[test]
    fn test_shader_features_cache_key_roundtrip() {
        let original = ShaderFeatures::new()
            .with_blend_animations(true)
            .with_smooth_lod(true)
            .with_debug_output(true);

        let key = original.cache_key();
        let restored = ShaderFeatures::from_cache_key(key);

        assert_eq!(original, restored);
    }

    #[test]
    fn test_shader_features_all_flags() {
        let features = ShaderFeatures {
            blend_animations: true,
            use_impostor: true,
            smooth_lod: true,
            debug_output: true,
            use_f16_textures: true,
            dual_quaternion_skinning: true,
        };

        let key = features.cache_key();
        assert_eq!(key, 0b111111);

        let restored = ShaderFeatures::from_cache_key(key);
        assert_eq!(features, restored);
    }

    // ========================================================================
    // ShaderConfig Tests
    // ========================================================================

    #[test]
    fn test_shader_config_default() {
        let config = ShaderConfig::default();
        assert_eq!(config.max_bones, 64);
        assert_eq!(config.sample_rate, 30.0);
        assert_eq!(config.workgroup_size, WORKGROUP_SIZE);
    }

    #[test]
    fn test_shader_config_builder() {
        let config = ShaderConfig::new()
            .with_max_bones(128)
            .with_sample_rate(60.0)
            .with_workgroup_size(128)
            .with_max_instances(5000);

        assert_eq!(config.max_bones, 128);
        assert_eq!(config.sample_rate, 60.0);
        assert_eq!(config.workgroup_size, 128);
        assert_eq!(config.max_instances, 5000);
    }

    #[test]
    fn test_shader_config_max_bones_clamped() {
        let config = ShaderConfig::new().with_max_bones(1000);
        assert_eq!(config.max_bones, MAX_BONES_FULL);
    }

    #[test]
    fn test_shader_config_sample_rate_clamped() {
        let config = ShaderConfig::new().with_sample_rate(0.0);
        assert_eq!(config.sample_rate, 1.0);
    }

    #[test]
    fn test_shader_config_workgroup_size_clamped() {
        let config1 = ShaderConfig::new().with_workgroup_size(0);
        assert_eq!(config1.workgroup_size, 1);

        let config2 = ShaderConfig::new().with_workgroup_size(1000);
        assert_eq!(config2.workgroup_size, 256);
    }

    #[test]
    fn test_shader_config_cache_key() {
        let config1 = ShaderConfig::default();
        let config2 = ShaderConfig::new().with_max_bones(128);

        assert_ne!(config1.cache_key(), config2.cache_key());
    }

    // ========================================================================
    // CrowdUniforms Tests
    // ========================================================================

    #[test]
    fn test_crowd_uniforms_size() {
        assert_eq!(std::mem::size_of::<CrowdUniforms>(), CROWD_UNIFORMS_SIZE);
    }

    #[test]
    fn test_crowd_uniforms_default() {
        let uniforms = CrowdUniforms::new();
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.bone_count, 64);
        assert_eq!(uniforms.sample_rate, 30.0);
    }

    #[test]
    fn test_crowd_uniforms_builder() {
        let uniforms = CrowdUniforms::new()
            .with_time(1.5, 0.016)
            .with_camera_pos([10.0, 5.0, -20.0])
            .with_animation(32, 60, 60.0)
            .with_lod_distances([15.0, 40.0, 80.0])
            .with_blend_weight(0.5);

        assert_eq!(uniforms.time, 1.5);
        assert_eq!(uniforms.delta_time, 0.016);
        assert_eq!(uniforms.camera_pos, [10.0, 5.0, -20.0]);
        assert_eq!(uniforms.bone_count, 32);
        assert_eq!(uniforms.frame_count, 60);
        assert_eq!(uniforms.sample_rate, 60.0);
        assert_eq!(uniforms.lod_distances, [15.0, 40.0, 80.0]);
        assert_eq!(uniforms.blend_weight, 0.5);
    }

    #[test]
    fn test_crowd_uniforms_blend_weight_clamped() {
        let u1 = CrowdUniforms::new().with_blend_weight(-0.5);
        assert_eq!(u1.blend_weight, 0.0);

        let u2 = CrowdUniforms::new().with_blend_weight(1.5);
        assert_eq!(u2.blend_weight, 1.0);
    }

    #[test]
    fn test_crowd_uniforms_as_bytes() {
        let uniforms = CrowdUniforms::new();
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), CROWD_UNIFORMS_SIZE);
    }

    // ========================================================================
    // ShaderStage Tests
    // ========================================================================

    #[test]
    fn test_shader_stage_flags() {
        let compute = ShaderStage::COMPUTE;
        assert!(compute.has_compute());
        assert!(!compute.has_vertex());
        assert!(!compute.has_fragment());

        let all = ShaderStage::ALL;
        assert!(all.has_compute());
        assert!(all.has_vertex());
        assert!(all.has_fragment());
    }

    #[test]
    fn test_shader_stage_bitor() {
        let combined = ShaderStage::VERTEX | ShaderStage::FRAGMENT;
        assert!(combined.has_vertex());
        assert!(combined.has_fragment());
        assert!(!combined.has_compute());
    }

    // ========================================================================
    // ShaderPermutation Tests
    // ========================================================================

    #[test]
    fn test_shader_permutation_cache_key() {
        let p1 = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());
        let p2 = ShaderPermutation::new(LodLevel::Simplified, ShaderFeatures::default());
        let p3 = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::new().with_blend_animations(true));

        assert_ne!(p1.cache_key(), p2.cache_key());
        assert_ne!(p1.cache_key(), p3.cache_key());
        assert_ne!(p2.cache_key(), p3.cache_key());
    }

    #[test]
    fn test_shader_permutation_same_key() {
        let p1 = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());
        let p2 = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        assert_eq!(p1.cache_key(), p2.cache_key());
    }

    // ========================================================================
    // ShaderCache Tests
    // ========================================================================

    #[test]
    fn test_shader_cache_new() {
        let cache = ShaderCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats(), (0, 0));
    }

    #[test]
    fn test_shader_cache_hit_miss() {
        let mut cache = ShaderCache::new();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        // First access - miss
        let _ = cache.get_or_insert(&perm, || "shader source".to_string());
        assert_eq!(cache.stats(), (0, 1));

        // Second access - hit
        let _ = cache.get_or_insert(&perm, || "should not be called".to_string());
        assert_eq!(cache.stats(), (1, 1));
    }

    #[test]
    fn test_shader_cache_contains() {
        let mut cache = ShaderCache::new();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        assert!(!cache.contains(&perm));

        let _ = cache.get_or_insert(&perm, || "source".to_string());
        assert!(cache.contains(&perm));
    }

    #[test]
    fn test_shader_cache_clear() {
        let mut cache = ShaderCache::new();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        let _ = cache.get_or_insert(&perm, || "source".to_string());
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats(), (0, 0));
    }

    // ========================================================================
    // CrowdAnimationShader Tests
    // ========================================================================

    #[test]
    fn test_crowd_shader_new() {
        let shader = CrowdAnimationShader::new(64, ShaderConfig::default());
        assert_eq!(shader.config().max_bones, 64);
    }

    #[test]
    fn test_crowd_shader_with_defaults() {
        let shader = CrowdAnimationShader::with_defaults(128);
        assert_eq!(shader.config().max_bones, 128);
    }

    #[test]
    fn test_crowd_shader_generate_source_lod0() {
        let shader = CrowdAnimationShader::default();
        let source = shader.generate_source(LodLevel::FullSkeleton);

        assert!(source.contains("TRINITY Crowd Animation Shader"));
        assert!(source.contains("#define LOD_LEVEL 0"));
        assert!(source.contains("sample_bone_transform"));
        assert!(source.contains("cs_skin_vertices"));
    }

    #[test]
    fn test_crowd_shader_generate_source_lod1() {
        let shader = CrowdAnimationShader::default();
        let source = shader.generate_source(LodLevel::Simplified);

        assert!(source.contains("#define LOD_LEVEL 1"));
        assert!(source.contains("cs_skin_vertices"));
    }

    #[test]
    fn test_crowd_shader_generate_source_lod2() {
        let shader = CrowdAnimationShader::default();
        let source = shader.generate_source(LodLevel::Impostor);

        assert!(source.contains("#define LOD_LEVEL 2"));
        assert!(source.contains("vs_impostor"));
        assert!(source.contains("fs_impostor"));
    }

    #[test]
    fn test_crowd_shader_generate_with_blend() {
        let features = ShaderFeatures::new().with_blend_animations(true);
        let shader = CrowdAnimationShader::new(64, ShaderConfig::default().with_features(features));
        let source = shader.generate_source(LodLevel::FullSkeleton);

        assert!(source.contains("sample_blended_bone_transform"));
        assert!(source.contains("mat_to_quat"));
    }

    #[test]
    fn test_crowd_shader_generate_without_blend() {
        let shader = CrowdAnimationShader::default();
        let source = shader.generate_source(LodLevel::FullSkeleton);

        // Should NOT contain blending code
        assert!(!source.contains("sample_blended_bone_transform"));
    }

    #[test]
    fn test_crowd_shader_bone_count_for_lod() {
        let shader = CrowdAnimationShader::new(64, ShaderConfig::default());

        assert_eq!(shader.bone_count_for_lod(LodLevel::FullSkeleton), 64);
        assert_eq!(shader.bone_count_for_lod(LodLevel::Simplified), 32);
        assert_eq!(shader.bone_count_for_lod(LodLevel::Impostor), 0);
    }

    #[test]
    fn test_crowd_shader_cache_permutation() {
        let mut shader = CrowdAnimationShader::default();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        // First generation
        let source1 = shader.generate_permutation(&perm).to_string();
        assert!(!source1.is_empty());
        assert_eq!(shader.cache_stats(), (0, 1));

        // Second access - should hit cache
        let source2 = shader.generate_permutation(&perm).to_string();
        assert_eq!(source1, source2);
        assert_eq!(shader.cache_stats(), (1, 1));
    }

    #[test]
    fn test_crowd_shader_is_cached() {
        let mut shader = CrowdAnimationShader::default();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        assert!(!shader.is_cached(&perm));

        let _ = shader.generate_permutation(&perm);
        assert!(shader.is_cached(&perm));
    }

    #[test]
    fn test_crowd_shader_clear_cache() {
        let mut shader = CrowdAnimationShader::default();
        let perm = ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default());

        let _ = shader.generate_permutation(&perm);
        assert!(shader.is_cached(&perm));

        shader.clear_cache();
        assert!(!shader.is_cached(&perm));
    }

    // ========================================================================
    // Bind Group Layout Tests
    // ========================================================================

    #[test]
    fn test_bind_group_layout_entries() {
        let shader = CrowdAnimationShader::default();
        let entries = shader.bind_group_layout_entries();

        assert_eq!(entries.len(), 6);

        // Check animation texture binding
        assert_eq!(entries[0].binding, BINDING_ANIMATION_TEXTURE);
        assert!(matches!(entries[0].ty, BindingType::Texture2D { .. }));

        // Check uniform buffer binding
        assert_eq!(entries[3].binding, BINDING_UNIFORMS);
        assert!(matches!(entries[3].ty, BindingType::UniformBuffer { .. }));
    }

    #[test]
    fn test_bind_group_layout_entries_blend() {
        let shader = CrowdAnimationShader::default();
        let entries = shader.bind_group_layout_entries_blend();

        // Should have one more entry for secondary animation texture
        assert_eq!(entries.len(), 7);
        assert_eq!(entries[6].binding, BINDING_ANIMATION_TEXTURE_B);
    }

    // ========================================================================
    // Workgroup Calculation Tests
    // ========================================================================

    #[test]
    fn test_workgroups_for_vertices() {
        let shader = CrowdAnimationShader::default();

        assert_eq!(shader.workgroups_for_vertices(64), 1);
        assert_eq!(shader.workgroups_for_vertices(65), 2);
        assert_eq!(shader.workgroups_for_vertices(128), 2);
        assert_eq!(shader.workgroups_for_vertices(1000), 16);
    }

    #[test]
    fn test_workgroups_for_instances() {
        let shader = CrowdAnimationShader::default();

        assert_eq!(shader.workgroups_for_instances(1), 1);
        assert_eq!(shader.workgroups_for_instances(100), 100);
    }

    // ========================================================================
    // Shader Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_source_valid() {
        let shader = CrowdAnimationShader::default();
        let source = shader.generate_source(LodLevel::FullSkeleton);

        assert!(shader.validate_source(&source).is_ok());
    }

    #[test]
    fn test_validate_source_missing_entry_point() {
        let shader = CrowdAnimationShader::default();
        let source = "fn helper() -> f32 { return 1.0; }";

        let result = shader.validate_source(source);
        assert!(matches!(result, Err(ShaderValidationError::MissingEntryPoint)));
    }

    #[test]
    fn test_validate_source_unbalanced_braces() {
        let shader = CrowdAnimationShader::default();
        let source = "@compute fn main() { { }";

        let result = shader.validate_source(source);
        assert!(matches!(result, Err(ShaderValidationError::UnbalancedBraces { .. })));
    }

    #[test]
    fn test_validate_source_unbalanced_parens() {
        let shader = CrowdAnimationShader::default();
        let source = "@compute fn main(a: u32 { }";

        let result = shader.validate_source(source);
        assert!(matches!(result, Err(ShaderValidationError::UnbalancedParentheses { .. })));
    }

    // ========================================================================
    // ShaderCompiler Tests
    // ========================================================================

    #[test]
    fn test_shader_compiler_preprocess() {
        let source = "fn main() {}";
        let defines = [("MAX_BONES", "64"), ("WORKGROUP_SIZE", "128")];

        let result = ShaderCompiler::preprocess(source, &defines);

        assert!(result.contains("#define MAX_BONES 64"));
        assert!(result.contains("#define WORKGROUP_SIZE 128"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_shader_compiler_strip_preprocessor() {
        let source = "#define FOO 1\nfn main() {}\n#ifdef BAR\n#endif";
        let result = ShaderCompiler::strip_preprocessor(source);

        assert!(!result.contains("#define"));
        assert!(!result.contains("#ifdef"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_shader_compiler_generate_skinning_shader() {
        let features = ShaderFeatures::default();
        let source = ShaderCompiler::generate_skinning_shader(64, &features);

        assert!(source.contains("cs_skin_vertices"));
        assert!(source.contains("#define MAX_BONES 64u"));
    }

    #[test]
    fn test_shader_compiler_line_count() {
        let source = "line1\nline2\nline3";
        assert_eq!(ShaderCompiler::line_count(source), 3);
    }

    #[test]
    fn test_shader_compiler_estimate_complexity() {
        let simple = "fn main() {}";
        let complex = "fn a() {} fn b() {} for (i in 0..10) { if (x) {} textureLoad() }";

        let simple_complexity = ShaderCompiler::estimate_complexity(simple);
        let complex_complexity = ShaderCompiler::estimate_complexity(complex);

        assert!(complex_complexity > simple_complexity);
    }

    // ========================================================================
    // Bilinear Sampling Tests
    // ========================================================================

    #[test]
    fn test_bilinear_sample_exact() {
        let data = vec![
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
        ];

        // Sample at exact pixel
        let result = bilinear_sample(&data, 2, 2, 0.0, 0.0);
        assert!((result[0] - 1.0).abs() < 0.0001);

        let result2 = bilinear_sample(&data, 2, 2, 1.0, 1.0);
        assert!((result2[0] - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_bilinear_sample_interpolated() {
        let data = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];

        // Sample at midpoint
        let result = bilinear_sample(&data, 2, 2, 0.5, 0.5);
        assert!((result[0] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_bilinear_sample_clamped() {
        let data = vec![[1.0, 2.0, 3.0, 4.0]];

        // Out of bounds should clamp
        let result = bilinear_sample(&data, 1, 1, -10.0, -10.0);
        assert!((result[0] - 1.0).abs() < 0.0001);

        let result2 = bilinear_sample(&data, 1, 1, 100.0, 100.0);
        assert!((result2[0] - 1.0).abs() < 0.0001);
    }

    // ========================================================================
    // Quaternion Interpolation Tests
    // ========================================================================

    #[test]
    fn test_nlerp_quat_identity() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let result = nlerp_quat(identity, identity, 0.5);

        assert!((result[3] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_nlerp_quat_endpoints() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let half_angle = PI / 4.0;
        let b = [0.0, half_angle.sin(), 0.0, half_angle.cos()];

        let result_0 = nlerp_quat(a, b, 0.0);
        let result_1 = nlerp_quat(a, b, 1.0);

        // At t=0, should be close to a
        assert!((result_0[3] - 1.0).abs() < 0.01);

        // At t=1, should be close to b
        let b_normalized = {
            let len = (b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3]).sqrt();
            [b[0]/len, b[1]/len, b[2]/len, b[3]/len]
        };
        for i in 0..4 {
            assert!((result_1[i] - b_normalized[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_nlerp_quat_normalized() {
        let a = [0.1, 0.2, 0.3, 0.9];
        let b = [0.4, 0.3, 0.2, 0.8];

        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = nlerp_quat(a, b, t);
            let len = (result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3]).sqrt();
            assert!((len - 1.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_nlerp_quat_antipodal() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0, 0.0, -1.0]; // Antipodal (same rotation)

        let result = nlerp_quat(a, b, 0.5);

        // Should still produce valid normalized quaternion
        let len = (result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3]).sqrt();
        assert!((len - 1.0).abs() < 0.0001);
    }

    // ========================================================================
    // Transform Composition Tests
    // ========================================================================

    #[test]
    fn test_compose_transform_identity() {
        let position = [0.0, 0.0, 0.0];
        let rotation = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
        let scale = 1.0;

        let m = compose_transform_matrix(position, rotation, scale);

        // Should be identity-ish
        assert!((m[0][0] - 1.0).abs() < 0.0001);
        assert!((m[1][1] - 1.0).abs() < 0.0001);
        assert!((m[2][2] - 1.0).abs() < 0.0001);
        assert!((m[3][3] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_compose_transform_position() {
        let position = [10.0, 20.0, 30.0];
        let rotation = [0.0, 0.0, 0.0, 1.0];
        let scale = 1.0;

        let m = compose_transform_matrix(position, rotation, scale);

        assert!((m[3][0] - 10.0).abs() < 0.0001);
        assert!((m[3][1] - 20.0).abs() < 0.0001);
        assert!((m[3][2] - 30.0).abs() < 0.0001);
    }

    #[test]
    fn test_compose_transform_scale() {
        let position = [0.0, 0.0, 0.0];
        let rotation = [0.0, 0.0, 0.0, 1.0];
        let scale = 2.0;

        let m = compose_transform_matrix(position, rotation, scale);

        assert!((m[0][0] - 2.0).abs() < 0.0001);
        assert!((m[1][1] - 2.0).abs() < 0.0001);
        assert!((m[2][2] - 2.0).abs() < 0.0001);
    }

    // ========================================================================
    // Linear Blend Skinning Tests
    // ========================================================================

    #[test]
    fn test_apply_lbs_identity() {
        let position = [1.0, 2.0, 3.0];
        let bone_indices = [0, 0, 0, 0];
        let bone_weights = [1.0, 0.0, 0.0, 0.0];
        let identity_matrix = compose_transform_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let bone_matrices = vec![identity_matrix];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        assert!((result[0] - 1.0).abs() < 0.0001);
        assert!((result[1] - 2.0).abs() < 0.0001);
        assert!((result[2] - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_apply_lbs_translation() {
        let position = [0.0, 0.0, 0.0];
        let bone_indices = [0, 0, 0, 0];
        let bone_weights = [1.0, 0.0, 0.0, 0.0];
        let translate_matrix = compose_transform_matrix([10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let bone_matrices = vec![translate_matrix];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        assert!((result[0] - 10.0).abs() < 0.0001);
        assert!((result[1] - 20.0).abs() < 0.0001);
        assert!((result[2] - 30.0).abs() < 0.0001);
    }

    #[test]
    fn test_apply_lbs_scale() {
        let position = [1.0, 1.0, 1.0];
        let bone_indices = [0, 0, 0, 0];
        let bone_weights = [1.0, 0.0, 0.0, 0.0];
        let scale_matrix = compose_transform_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 2.0);
        let bone_matrices = vec![scale_matrix];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        assert!((result[0] - 2.0).abs() < 0.0001);
        assert!((result[1] - 2.0).abs() < 0.0001);
        assert!((result[2] - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_apply_lbs_multi_bone() {
        let position = [0.0, 0.0, 0.0];
        let bone_indices = [0, 1, 0, 0];
        let bone_weights = [0.5, 0.5, 0.0, 0.0];

        let bone0 = compose_transform_matrix([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let bone1 = compose_transform_matrix([0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let bone_matrices = vec![bone0, bone1];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        // Should be average of (10, 0, 0) and (0, 10, 0) = (5, 5, 0)
        assert!((result[0] - 5.0).abs() < 0.0001);
        assert!((result[1] - 5.0).abs() < 0.0001);
        assert!((result[2] - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_apply_lbs_zero_weights() {
        let position = [1.0, 2.0, 3.0];
        let bone_indices = [0, 1, 2, 3];
        let bone_weights = [0.0, 0.0, 0.0, 0.0];

        let bone_matrices = vec![
            compose_transform_matrix([100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0),
        ];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        // With zero weights, result should be zero
        assert!((result[0] - 0.0).abs() < 0.0001);
        assert!((result[1] - 0.0).abs() < 0.0001);
        assert!((result[2] - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_apply_lbs_out_of_bounds_bone() {
        let position = [1.0, 1.0, 1.0];
        let bone_indices = [100, 0, 0, 0]; // Index 100 doesn't exist
        let bone_weights = [1.0, 0.0, 0.0, 0.0];
        let bone_matrices = vec![
            compose_transform_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0),
        ];

        let result = apply_lbs(position, bone_indices, bone_weights, &bone_matrices);

        // Out of bounds should not crash, returns zero contribution
        assert!((result[0] - 0.0).abs() < 0.0001);
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_shader_validation_error_display() {
        let err = ShaderValidationError::MissingEntryPoint;
        assert!(format!("{}", err).contains("entry point"));

        let err2 = ShaderValidationError::UnbalancedBraces { open: 5, close: 3 };
        assert!(format!("{}", err2).contains("5"));
        assert!(format!("{}", err2).contains("3"));
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_shader_pipeline() {
        // Create shader manager
        let config = ShaderConfig::new()
            .with_max_bones(64)
            .with_features(ShaderFeatures::new().with_blend_animations(true));
        let shader = CrowdAnimationShader::new(64, config);

        // Generate all LOD variants
        for lod in LodLevel::all() {
            let source = shader.generate_source(lod);

            // Validate each variant
            let result = shader.validate_source(&source);
            assert!(result.is_ok(), "LOD {:?} failed validation: {:?}", lod, result);

            // Check that source is non-empty
            assert!(!source.is_empty());

            // Check for LOD-specific content
            match lod {
                LodLevel::FullSkeleton | LodLevel::Simplified => {
                    assert!(source.contains("cs_skin_vertices"));
                }
                LodLevel::Impostor => {
                    assert!(source.contains("vs_impostor"));
                }
            }
        }
    }

    #[test]
    fn test_permutation_caching_all_variants() {
        let mut shader = CrowdAnimationShader::default();

        // Generate multiple permutations
        let permutations = vec![
            ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::default()),
            ShaderPermutation::new(LodLevel::Simplified, ShaderFeatures::default()),
            ShaderPermutation::new(LodLevel::Impostor, ShaderFeatures::default()),
            ShaderPermutation::new(LodLevel::FullSkeleton, ShaderFeatures::new().with_blend_animations(true)),
        ];

        // First pass - all misses
        for perm in &permutations {
            let _ = shader.generate_permutation(perm);
        }
        assert_eq!(shader.cache_stats(), (0, 4));

        // Second pass - all hits
        for perm in &permutations {
            let _ = shader.generate_permutation(perm);
        }
        assert_eq!(shader.cache_stats(), (4, 4));
    }

    #[test]
    fn test_uniforms_roundtrip() {
        let uniforms = CrowdUniforms::new()
            .with_time(2.5, 0.016)
            .with_camera_pos([100.0, 50.0, -200.0])
            .with_animation(128, 90, 30.0)
            .with_lod_distances([10.0, 30.0, 60.0])
            .with_blend_weight(0.75)
            .with_texture_dimensions(512, 256);

        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), CROWD_UNIFORMS_SIZE);

        // Reconstruct from bytes
        let reconstructed: CrowdUniforms = *bytemuck::from_bytes(bytes);

        assert_eq!(reconstructed.time, uniforms.time);
        assert_eq!(reconstructed.bone_count, uniforms.bone_count);
        assert_eq!(reconstructed.camera_pos, uniforms.camera_pos);
        assert_eq!(reconstructed.blend_weight, uniforms.blend_weight);
    }
}
