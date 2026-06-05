// SPDX-License-Identifier: MIT
//
// gpu_particle_spawn.comp.wgsl - GPU Particle Spawning for TRINITY Engine (T-GPU-5.1)
//
// Spawns new particles from emitter configuration using PCG-based random generation.
// Uses indirect spawn count for variable spawn rates, allowing CPU or compute shader
// to control the exact number of particles to spawn each frame.
//
// Algorithm:
// 1. Each thread reads its spawn index from global_invocation_id
// 2. Check if index < spawn_count (indirect parameter)
// 3. Generate random values using PCG hash seeded with time + index
// 4. Initialize particle at offset position within emitter sphere
// 5. Set random velocity within configured range
// 6. Set lifetime, size, color, and rotation
//
// Performance:
// - Workgroup size 256 for optimal GPU occupancy
// - Each thread spawns exactly one particle (if index < spawn_count)
// - Memory-coalesced writes to particle buffer

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters controlling particle spawning for this frame.
struct ParticleSpawnParams {
    /// Number of particles to spawn this frame.
    spawn_count: u32,
    /// Write offset in the particle buffer (where to start writing).
    particle_offset: u32,
    /// Maximum particles the buffer can hold.
    max_particles: u32,
    /// Current simulation time (seconds).
    time: f32,
    /// Delta time since last frame (seconds).
    delta_time: f32,
    /// Random seed offset for this frame.
    random_seed: u32,
    /// Padding for 16-byte alignment.
    _padding: vec2<u32>,
}

/// Emitter configuration defining spawn characteristics.
struct EmitterConfig {
    /// World-space position of emitter center.
    position: vec3<f32>,
    /// Radius of spherical spawn region.
    spawn_radius: f32,

    /// Minimum initial velocity (per axis).
    velocity_min: vec3<f32>,
    /// Velocity spread factor (randomness).
    velocity_spread: f32,

    /// Maximum initial velocity (per axis).
    velocity_max: vec3<f32>,
    /// Minimum particle lifetime (seconds).
    lifetime_min: f32,

    /// Starting color (RGBA premultiplied alpha).
    color_start: vec4<f32>,

    /// Ending color (RGBA premultiplied alpha, interpolated over lifetime).
    color_end: vec4<f32>,

    /// Starting particle size (world units).
    size_start: f32,
    /// Ending particle size (world units).
    size_end: f32,
    /// Maximum particle lifetime (seconds).
    lifetime_max: f32,
    /// Maximum rotation speed (radians/second).
    rotation_speed_max: f32,
}

/// GPU particle data (SoA-friendly layout, 64 bytes per particle).
/// Aligned for efficient GPU memory access patterns.
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

@group(0) @binding(0) var<uniform> params: ParticleSpawnParams;
@group(0) @binding(1) var<uniform> emitter: EmitterConfig;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

// ============================================================================
// PCG Random Number Generator
// ============================================================================
// PCG (Permuted Congruential Generator) provides high-quality randomness
// suitable for visual effects while being fast on GPU.

/// PCG hash function - produces well-distributed 32-bit output from 32-bit seed.
fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Generate a random float in [0, 1) from a mutable seed.
fn random_float(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967296.0;
}

/// Generate a random float in [min, max) from a mutable seed.
fn random_range(seed: ptr<function, u32>, min_val: f32, max_val: f32) -> f32 {
    return min_val + random_float(seed) * (max_val - min_val);
}

/// Generate a random unit vector on the surface of a sphere.
fn random_on_sphere(seed: ptr<function, u32>) -> vec3<f32> {
    let theta = random_float(seed) * TWO_PI;
    let phi = acos(2.0 * random_float(seed) - 1.0);
    let sin_phi = sin(phi);
    return vec3<f32>(
        sin_phi * cos(theta),
        sin_phi * sin(theta),
        cos(phi)
    );
}

/// Generate a random point inside a sphere of given radius.
fn random_in_sphere(seed: ptr<function, u32>, radius: f32) -> vec3<f32> {
    // Use cube root of uniform random for uniform distribution inside sphere
    let r = radius * pow(random_float(seed), 1.0 / 3.0);
    return random_on_sphere(seed) * r;
}

/// Generate a random direction in a hemisphere oriented along +Y.
fn random_in_hemisphere_y(seed: ptr<function, u32>) -> vec3<f32> {
    let dir = random_on_sphere(seed);
    // Flip to positive Y hemisphere
    return vec3<f32>(dir.x, abs(dir.y), dir.z);
}

/// Generate a random vec3 where each component is independently randomized.
fn random_vec3_range(
    seed: ptr<function, u32>,
    min_val: vec3<f32>,
    max_val: vec3<f32>
) -> vec3<f32> {
    return vec3<f32>(
        random_range(seed, min_val.x, max_val.x),
        random_range(seed, min_val.y, max_val.y),
        random_range(seed, min_val.z, max_val.z)
    );
}

// ============================================================================
// Main Entry Point
// ============================================================================

@compute @workgroup_size(256)
fn spawn_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_index = gid.x;

    // Bounds check: only spawn if within spawn_count for this frame
    if (spawn_index >= params.spawn_count) {
        return;
    }

    // Calculate write index in particle buffer
    let particle_index = params.particle_offset + spawn_index;

    // Check buffer capacity (should not exceed max_particles)
    if (particle_index >= params.max_particles) {
        return;
    }

    // Initialize random seed from time, index, and frame seed
    // Use bit mixing to ensure good distribution across threads
    var seed: u32 = pcg_hash(params.random_seed + spawn_index);
    seed = pcg_hash(seed ^ bitcast<u32>(params.time));
    seed = pcg_hash(seed + particle_index);

    // Generate spawn position within emitter's spawn radius
    let offset = random_in_sphere(&seed, emitter.spawn_radius);
    let position = emitter.position + offset;

    // Generate initial velocity
    // Interpolate between min and max with spread-based randomization
    let base_velocity = (emitter.velocity_min + emitter.velocity_max) * 0.5;
    let velocity_range = (emitter.velocity_max - emitter.velocity_min) * 0.5;

    // Add directional component with spread
    let spread_dir = random_on_sphere(&seed) * emitter.velocity_spread;
    let velocity = base_velocity + velocity_range * spread_dir +
        random_vec3_range(&seed, -velocity_range, velocity_range);

    // Generate lifetime
    let lifetime = random_range(&seed, emitter.lifetime_min, emitter.lifetime_max);

    // Initial size (interpolated at age=0 means start size)
    let size = emitter.size_start;

    // Initial color (start color at age=0)
    let color = emitter.color_start;

    // Initial rotation (random starting angle)
    let rotation = random_float(&seed) * TWO_PI;

    // Rotation speed (random within configured range)
    let rotation_speed = random_range(&seed, -emitter.rotation_speed_max, emitter.rotation_speed_max);

    // Write particle to buffer
    particles[particle_index] = Particle(
        position,
        0.0,            // age = 0 (just spawned)
        velocity,
        lifetime,
        color,
        size,
        rotation,
        rotation_speed,
        1u              // flags: bit 0 = alive
    );
}

// ============================================================================
// Variant: Spawn with Direction Bias
// ============================================================================
// This variant spawns particles with a bias toward a specific direction,
// useful for directional emitters like flames, sparks, or water jets.

@compute @workgroup_size(256)
fn spawn_particles_directed(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let spawn_index = gid.x;

    if (spawn_index >= params.spawn_count) {
        return;
    }

    let particle_index = params.particle_offset + spawn_index;

    if (particle_index >= params.max_particles) {
        return;
    }

    var seed: u32 = pcg_hash(params.random_seed + spawn_index);
    seed = pcg_hash(seed ^ bitcast<u32>(params.time));

    // Position with spawn radius
    let offset = random_in_sphere(&seed, emitter.spawn_radius);
    let position = emitter.position + offset;

    // For directed emitters, use velocity_min as base direction
    // and velocity_max as magnitude range
    let base_dir = normalize(emitter.velocity_max - emitter.velocity_min);
    let speed = length(emitter.velocity_max);

    // Add cone spread around base direction
    let spread_angle = emitter.velocity_spread * PI * 0.5;
    let cone_random = random_in_hemisphere_y(&seed);

    // Rotate cone_random to align with base_dir
    // Simple approach: create basis from base_dir
    var tangent: vec3<f32>;
    if (abs(base_dir.y) < 0.9) {
        tangent = normalize(cross(base_dir, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        tangent = normalize(cross(base_dir, vec3<f32>(1.0, 0.0, 0.0)));
    }
    let bitangent = cross(base_dir, tangent);

    // Construct direction in cone
    let theta_spread = random_float(&seed) * spread_angle;
    let phi_spread = random_float(&seed) * TWO_PI;
    let cone_offset = tangent * sin(theta_spread) * cos(phi_spread) +
        bitangent * sin(theta_spread) * sin(phi_spread);
    let final_dir = normalize(base_dir + cone_offset);

    // Randomize speed
    let min_speed = length(emitter.velocity_min);
    let final_speed = random_range(&seed, min_speed, speed);
    let velocity = final_dir * final_speed;

    // Rest is same as basic spawn
    let lifetime = random_range(&seed, emitter.lifetime_min, emitter.lifetime_max);
    let rotation = random_float(&seed) * TWO_PI;
    let rotation_speed = random_range(&seed, -emitter.rotation_speed_max, emitter.rotation_speed_max);

    particles[particle_index] = Particle(
        position,
        0.0,
        velocity,
        lifetime,
        emitter.color_start,
        emitter.size_start,
        rotation,
        rotation_speed,
        1u
    );
}
