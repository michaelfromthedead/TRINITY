// SPDX-License-Identifier: MIT
//
// mesh_particle.vert.wgsl - Mesh Particle Vertex Shader (T-GPU-6.2)
//
// Renders mesh particles at particle positions with scaling and rotation.
// Each particle instance has its own transform derived from particle data.
//
// Features:
// - Instanced rendering: one mesh drawn at each particle position
// - Scale from particle size or uniform scale
// - Rotation around velocity axis or Y-up
// - Color modulation from particle color
// - Normal transformation for proper lighting

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters controlling mesh particle rendering.
struct MeshParticleParams {
    /// Combined view-projection matrix.
    view_proj: mat4x4<f32>,
    /// 0 = uniform scale, 1 = scale from particle size.
    scale_from_size: u32,
    /// Base scale when scale_from_size is 0.
    base_scale: f32,
    /// Rotation mode: 0 = Y-up, 1 = velocity aligned.
    rotation_mode: u32,
    /// Padding for 16-byte alignment.
    _pad: u32,
}

/// GPU particle data (64 bytes, matches spawn.rs Particle struct).
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

/// Vertex input from mesh geometry.
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

/// Vertex output to fragment shader.
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: MeshParticleParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> sort_indices: array<u32>;

// ============================================================================
// Rotation Matrix Functions
// ============================================================================

/// Build a rotation matrix around the Y axis.
fn rotation_matrix_y(angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3<f32>(
        vec3<f32>(c, 0.0, -s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(s, 0.0, c)
    );
}

/// Build a rotation matrix that aligns Y-up with the given direction.
/// Returns identity if direction is too small.
fn align_to_velocity(velocity: vec3<f32>) -> mat3x3<f32> {
    let speed = length(velocity);
    if speed < 0.001 {
        // No meaningful velocity, use identity
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0)
        );
    }

    let forward = normalize(velocity);

    // Choose an up vector that is not parallel to forward
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(dot(forward, up)) > 0.99 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }

    // Construct orthonormal basis (Gram-Schmidt)
    let right = normalize(cross(up, forward));
    let corrected_up = cross(forward, right);

    // Return rotation matrix (columns are basis vectors)
    return mat3x3<f32>(right, corrected_up, forward);
}

/// Build the rotation matrix based on rotation mode and particle data.
fn build_rotation_matrix(particle: Particle, rotation_mode: u32) -> mat3x3<f32> {
    if rotation_mode == 1u {
        // Velocity-aligned rotation
        let velocity_rotation = align_to_velocity(particle.velocity);
        // Apply additional rotation around the aligned forward axis
        let angle_rotation = rotation_matrix_y(particle.rotation);
        return velocity_rotation * angle_rotation;
    } else {
        // Y-up rotation (default)
        return rotation_matrix_y(particle.rotation);
    }
}

// ============================================================================
// Vertex Shader Entry Point
// ============================================================================

@vertex
fn vs_mesh_particle(
    vertex: VertexInput,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    var output: VertexOutput;

    // Look up the particle index (sorted order for correct alpha blending)
    let particle_index = sort_indices[instance_id];
    let particle = particles[particle_index];

    // Skip dead particles by rendering at origin with zero scale
    // (GPU will cull degenerate triangles)
    if (particle.flags & 1u) == 0u {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        output.uv = vec2<f32>(0.0, 0.0);
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }

    // Determine scale
    var scale: f32;
    if params.scale_from_size == 1u {
        scale = particle.size;
    } else {
        scale = params.base_scale;
    }

    // Build rotation matrix
    let rotation = build_rotation_matrix(particle, params.rotation_mode);

    // Transform vertex position:
    // 1. Scale the vertex
    // 2. Rotate around particle's rotation axis
    // 3. Translate to particle position
    let scaled_pos = vertex.position * scale;
    let rotated_pos = rotation * scaled_pos;
    let world_pos = rotated_pos + particle.position;

    // Transform to clip space
    output.clip_position = params.view_proj * vec4<f32>(world_pos, 1.0);

    // Transform normal (rotation only, no scale for uniform scaling)
    output.world_normal = normalize(rotation * vertex.normal);

    // Pass through UV
    output.uv = vertex.uv;

    // Pass through particle color
    output.color = particle.color;

    return output;
}
