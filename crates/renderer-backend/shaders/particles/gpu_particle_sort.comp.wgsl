// SPDX-License-Identifier: MIT
//
// gpu_particle_sort.comp.wgsl - GPU Particle Depth Sort for TRINITY Engine (T-GPU-5.4)
//
// Computes depth keys for particles and builds an indirection array for sorted
// rendering. Uses the GpuRadixSort infrastructure from gpu_driven for the actual
// sorting - this shader only computes the depth keys.
//
// For correct alpha blending, transparent particles must be rendered back-to-front.
// This shader computes depth as distance along the view direction and creates
// a sort key that places far particles first (low keys) and near particles last.
//
// Algorithm:
// 1. Each thread computes the depth of one particle along view direction
// 2. Depth is quantized to 32-bit key with far = low (back-to-front ordering)
// 3. Indices are initialized for indirection array
// 4. GpuRadixSort then sorts keys+indices for final ordering
//
// Performance:
// - Workgroup size 256 for optimal GPU occupancy
// - Single-pass key computation before radix sort
// - Output is indirection array (particle buffer unchanged)

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const PARTICLE_FLAG_ALIVE: u32 = 1u;

// Depth quantization parameters
// We use float-to-uint bitcast and flip for back-to-front ordering
const DEPTH_QUANTIZATION_SCALE: f32 = 16777216.0; // 2^24 for precision

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters for particle depth sorting.
struct ParticleSortParams {
    /// Number of particles to process.
    num_particles: u32,
    /// Padding for 16-byte alignment of vec3.
    _pad0: u32,
    /// Padding for 16-byte alignment of vec3.
    _pad1: u32,
    /// Padding for 16-byte alignment of vec3.
    _pad2: u32,

    /// Camera world-space position.
    camera_position: vec3<f32>,
    /// Near plane distance (for depth normalization).
    near_plane: f32,

    /// Camera view direction (normalized, points into scene).
    view_direction: vec3<f32>,
    /// Far plane distance (for depth normalization).
    far_plane: f32,
}

/// GPU particle data (matches spawn.rs Particle struct).
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

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: ParticleSortParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> sort_indices: array<u32>;

// ============================================================================
// Depth Key Computation
// ============================================================================

/// Compute depth key for a particle position.
/// Far particles get low keys, near particles get high keys (back-to-front).
///
/// Returns 0xFFFFFFFF for dead particles (sorts to end).
fn compute_depth_key(
    pos: vec3<f32>,
    camera_pos: vec3<f32>,
    view_dir: vec3<f32>,
    near: f32,
    far: f32
) -> u32 {
    // Compute signed distance along view direction
    let to_particle = pos - camera_pos;
    let depth = dot(to_particle, view_dir);

    // Clamp to valid range
    let clamped_depth = clamp(depth, near, far);

    // Normalize to [0, 1] range
    let depth_range = far - near;
    let normalized = (clamped_depth - near) / depth_range;

    // Quantize to 32-bit integer
    // Flip bits so far (high normalized) becomes low key (sorted first)
    let quantized = u32(normalized * DEPTH_QUANTIZATION_SCALE);
    return 0xFFFFFFu - quantized; // Back-to-front: far particles have lower keys
}

/// Compute depth key with float-to-bits method for higher precision.
/// This preserves floating point ordering properties.
fn compute_depth_key_precise(
    pos: vec3<f32>,
    camera_pos: vec3<f32>,
    view_dir: vec3<f32>
) -> u32 {
    // Compute signed distance along view direction
    let to_particle = pos - camera_pos;
    let depth = dot(to_particle, view_dir);

    // Convert float to sortable uint:
    // - Positive floats: flip sign bit (0x80000000 ^ bits) -> larger float = larger uint
    // - For back-to-front, we want far (larger depth) to sort first (smaller key)
    // So we flip all bits to reverse the order
    let bits = bitcast<u32>(depth);

    // Handle negative depths (behind camera) - push to very end
    if (depth < 0.0) {
        return 0xFFFFFFFFu;
    }

    // Flip sign bit for proper ordering of positive floats
    let sortable = bits ^ 0x80000000u;

    // Flip all bits for back-to-front (far = low key)
    return 0xFFFFFFFFu - sortable;
}

// ============================================================================
// Main Entry Point: Compute Sort Keys
// ============================================================================

@compute @workgroup_size(256)
fn compute_sort_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= params.num_particles) {
        return;
    }

    let particle = particles[idx];

    // Initialize index for indirection array
    sort_indices[idx] = idx;

    // Check if particle is alive
    if ((particle.flags & PARTICLE_FLAG_ALIVE) == 0u) {
        // Dead particles get maximum key (sort to end, won't be rendered)
        sort_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    // Compute depth key using precise method
    sort_keys[idx] = compute_depth_key_precise(
        particle.position,
        params.camera_position,
        params.view_direction
    );
}

// ============================================================================
// Variant: Compute Sort Keys with Normalized Depth
// ============================================================================
// This variant uses the near/far plane parameters for explicit depth normalization.
// Useful when you need predictable depth ranges.

@compute @workgroup_size(256)
fn compute_sort_keys_normalized(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_particles) {
        return;
    }

    let particle = particles[idx];
    sort_indices[idx] = idx;

    if ((particle.flags & PARTICLE_FLAG_ALIVE) == 0u) {
        sort_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    sort_keys[idx] = compute_depth_key(
        particle.position,
        params.camera_position,
        params.view_direction,
        params.near_plane,
        params.far_plane
    );
}

// ============================================================================
// Variant: Compute Sort Keys with Distance (Radial)
// ============================================================================
// This variant sorts by distance from camera rather than depth along view axis.
// Useful for omnidirectional effects or when particles can be behind the camera.

@compute @workgroup_size(256)
fn compute_sort_keys_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_particles) {
        return;
    }

    let particle = particles[idx];
    sort_indices[idx] = idx;

    if ((particle.flags & PARTICLE_FLAG_ALIVE) == 0u) {
        sort_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    // Compute distance from camera
    let to_particle = particle.position - params.camera_position;
    let distance = length(to_particle);

    // Convert to sortable key (far first)
    let bits = bitcast<u32>(distance);
    let sortable = bits ^ 0x80000000u;
    sort_keys[idx] = 0xFFFFFFFFu - sortable;
}
