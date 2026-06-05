// SPDX-License-Identifier: MIT
//
// billboard.vert.wgsl - Billboard Particle Vertex Shader (T-GPU-6.1)
//
// Generates camera-facing billboard quads for particle rendering.
// Each particle is expanded into a 4-vertex quad using triangle strip topology.
//
// Supports three alignment modes:
// - VIEW: Billboard faces camera (standard billboarding)
// - VELOCITY: Billboard stretches along velocity vector (smoke trails)
// - CUSTOM: Billboard uses custom axis (special effects)
//
// Algorithm:
// 1. Read particle data from sorted index array (back-to-front order)
// 2. Calculate billboard axes based on alignment mode
// 3. Apply particle rotation around billboard normal
// 4. Scale by particle size and velocity stretch (for VELOCITY mode)
// 5. Transform to clip space

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Alignment modes
const ALIGN_VIEW: u32 = 0u;
const ALIGN_VELOCITY: u32 = 1u;
const ALIGN_CUSTOM: u32 = 2u;

// Quad vertex offsets (triangle strip: BL, BR, TL, TR)
// Forms two triangles: (BL, BR, TL) and (BR, TR, TL)
const QUAD_OFFSETS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, -0.5),  // 0: Bottom-left
    vec2<f32>( 0.5, -0.5),  // 1: Bottom-right
    vec2<f32>(-0.5,  0.5),  // 2: Top-left
    vec2<f32>( 0.5,  0.5)   // 3: Top-right
);

// UV coordinates matching quad offsets
const QUAD_UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),  // 0: Bottom-left
    vec2<f32>(1.0, 1.0),  // 1: Bottom-right
    vec2<f32>(0.0, 0.0),  // 2: Top-left
    vec2<f32>(1.0, 0.0)   // 3: Top-right
);

// ============================================================================
// Data Structures
// ============================================================================

/// Uniform parameters for billboard rendering.
struct BillboardParams {
    /// View matrix (world to camera space).
    view_matrix: mat4x4<f32>,
    /// Projection matrix (camera to clip space).
    proj_matrix: mat4x4<f32>,
    /// Camera right vector in world space (for VIEW alignment).
    camera_right: vec3<f32>,
    /// Alignment mode: 0=VIEW, 1=VELOCITY, 2=CUSTOM.
    alignment_mode: u32,
    /// Camera up vector in world space (for VIEW alignment).
    camera_up: vec3<f32>,
    /// Velocity stretch factor for VELOCITY alignment (1.0 = no stretch).
    velocity_stretch: f32,
    /// Custom axis for CUSTOM alignment mode.
    custom_axis: vec3<f32>,
    /// Time for animated effects.
    time: f32,
}

/// GPU particle data (matches spawn.rs Particle struct - 64 bytes).
struct Particle {
    /// World-space position.
    position: vec3<f32>,
    /// Current age (seconds since spawn).
    age: f32,

    /// Current velocity (world units/second).
    velocity: vec3<f32>,
    /// Total lifetime (seconds).
    lifetime: f32,

    /// Current color (RGBA premultiplied alpha).
    color: vec4<f32>,

    /// Current size (world units).
    size: f32,
    /// Current rotation (radians).
    rotation: f32,
    /// Rotation speed (radians/second).
    rotation_speed: f32,
    /// Flags (bit 0: alive).
    flags: u32,
}

/// Vertex shader output to fragment shader.
struct VertexOutput {
    /// Clip-space position (required by GPU).
    @builtin(position) clip_position: vec4<f32>,
    /// Texture coordinates for particle texture.
    @location(0) uv: vec2<f32>,
    /// Particle color (interpolated over lifetime).
    @location(1) color: vec4<f32>,
    /// Normalized age (0=just spawned, 1=about to die).
    @location(2) age_ratio: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: BillboardParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> sort_indices: array<u32>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Rotate a 2D vector by the given angle (radians).
fn rotate_2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    );
}

/// Calculate billboard axes for VIEW alignment (camera-facing).
fn calculate_view_axes(
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    rotation: f32
) -> array<vec3<f32>, 2> {
    // Apply rotation around view direction (cross of right and up)
    let c = cos(rotation);
    let s = sin(rotation);

    let right = camera_right * c + camera_up * s;
    let up = camera_up * c - camera_right * s;

    return array<vec3<f32>, 2>(right, up);
}

/// Calculate billboard axes for VELOCITY alignment (stretches along velocity).
fn calculate_velocity_axes(
    velocity: vec3<f32>,
    camera_pos: vec3<f32>,
    particle_pos: vec3<f32>,
    stretch_factor: f32,
    rotation: f32
) -> array<vec3<f32>, 2> {
    let speed = length(velocity);

    // If velocity is near zero, fall back to camera-facing
    if speed < 0.001 {
        // Get view direction and construct axes
        let view_dir = normalize(particle_pos - camera_pos);
        let world_up = vec3<f32>(0.0, 1.0, 0.0);
        let right = normalize(cross(world_up, view_dir));
        let up = normalize(cross(view_dir, right));
        return array<vec3<f32>, 2>(right, up);
    }

    // Velocity direction is the "up" axis for the billboard
    let vel_dir = velocity / speed;

    // Get view direction from particle to camera
    let to_camera = normalize(camera_pos - particle_pos);

    // Right axis is perpendicular to both velocity and view
    var right = normalize(cross(vel_dir, to_camera));

    // Handle degenerate case where velocity is parallel to view
    if length(right) < 0.001 {
        let world_up = vec3<f32>(0.0, 1.0, 0.0);
        right = normalize(cross(vel_dir, world_up));
    }

    // Up axis aligns with velocity, stretched by factor
    let up = vel_dir * (1.0 + speed * stretch_factor);

    // Apply rotation around velocity axis
    let c = cos(rotation);
    let s = sin(rotation);
    let rotated_right = right * c + cross(vel_dir, right) * s;

    return array<vec3<f32>, 2>(rotated_right, up);
}

/// Calculate billboard axes for CUSTOM alignment.
fn calculate_custom_axes(
    custom_axis: vec3<f32>,
    camera_pos: vec3<f32>,
    particle_pos: vec3<f32>,
    rotation: f32
) -> array<vec3<f32>, 2> {
    // Custom axis is the constrained axis (billboard rotates around it)
    let axis = normalize(custom_axis);

    // Get view direction
    let to_camera = normalize(camera_pos - particle_pos);

    // Right axis is perpendicular to custom axis and view
    var right = normalize(cross(axis, to_camera));

    // Handle degenerate case
    if length(right) < 0.001 {
        let world_right = vec3<f32>(1.0, 0.0, 0.0);
        right = normalize(cross(axis, world_right));
    }

    // Apply rotation
    let c = cos(rotation);
    let s = sin(rotation);
    let rotated_right = right * c + cross(axis, right) * s;
    let up = axis;

    return array<vec3<f32>, 2>(rotated_right, up);
}

// ============================================================================
// Vertex Shader Entry Point
// ============================================================================

@vertex
fn vs_billboard(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    var output: VertexOutput;

    // Get particle index from sorted array (for back-to-front rendering)
    let particle_index = sort_indices[instance_id];
    let particle = particles[particle_index];

    // Check if particle is alive (bit 0 of flags)
    let is_alive = (particle.flags & 1u) != 0u;
    if !is_alive {
        // Dead particle: output degenerate triangle (will be clipped)
        output.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        output.uv = vec2<f32>(0.0, 0.0);
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.age_ratio = 1.0;
        return output;
    }

    // Get quad vertex (0-3 for triangle strip)
    let quad_vertex = vertex_id % 4u;
    let offset = QUAD_OFFSETS[quad_vertex];

    // Calculate normalized age
    var age_ratio = 0.0;
    if particle.lifetime > 0.0 {
        age_ratio = clamp(particle.age / particle.lifetime, 0.0, 1.0);
    }

    // Extract camera position from inverse view matrix
    let inv_view = params.view_matrix;
    let camera_pos = vec3<f32>(
        -dot(inv_view[0].xyz, inv_view[3].xyz),
        -dot(inv_view[1].xyz, inv_view[3].xyz),
        -dot(inv_view[2].xyz, inv_view[3].xyz)
    );

    // Calculate billboard axes based on alignment mode
    var axes: array<vec3<f32>, 2>;

    switch params.alignment_mode {
        case ALIGN_VIEW: {
            axes = calculate_view_axes(
                params.camera_right,
                params.camera_up,
                particle.rotation
            );
        }
        case ALIGN_VELOCITY: {
            axes = calculate_velocity_axes(
                particle.velocity,
                camera_pos,
                particle.position,
                params.velocity_stretch,
                particle.rotation
            );
        }
        case ALIGN_CUSTOM: {
            axes = calculate_custom_axes(
                params.custom_axis,
                camera_pos,
                particle.position,
                particle.rotation
            );
        }
        default: {
            // Fallback to view alignment
            axes = calculate_view_axes(
                params.camera_right,
                params.camera_up,
                particle.rotation
            );
        }
    }

    let right = axes[0];
    let up = axes[1];

    // Calculate world position of this vertex
    let world_offset = right * offset.x * particle.size + up * offset.y * particle.size;
    let world_position = particle.position + world_offset;

    // Transform to clip space
    let view_position = params.view_matrix * vec4<f32>(world_position, 1.0);
    output.clip_position = params.proj_matrix * view_position;

    // Pass through UV and color
    output.uv = QUAD_UVS[quad_vertex];
    output.color = particle.color;
    output.age_ratio = age_ratio;

    return output;
}
