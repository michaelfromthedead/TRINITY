// SPDX-License-Identifier: MIT
//
// gpu_particle_update.comp.wgsl - GPU Particle Update for TRINITY Engine (T-GPU-5.2)
//
// Updates particle physics each frame: position, velocity, age, color, size.
// Supports multiple force types: gravity, wind, turbulence, vortex, attraction.
// Uses ping-pong buffers: reads from A, writes to B, then swap.
//
// Algorithm:
// 1. Each thread reads one particle from input buffer
// 2. Check if particle is alive (age < lifetime and flags bit 0 set)
// 3. Advance age by delta_time
// 4. If age >= lifetime, mark as dead in alive_flags
// 5. Apply all configured forces to velocity
// 6. Apply drag to velocity
// 7. Integrate position: position += velocity * dt
// 8. Interpolate color and size based on age/lifetime ratio
// 9. Advance rotation by rotation_speed * dt
// 10. Write updated particle to output buffer
//
// Performance:
// - Workgroup size 256 for optimal GPU occupancy
// - Coalesced memory access patterns
// - No branching in force calculation (masked addition)

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

/// Flag bit indicating particle is alive.
const FLAG_ALIVE: u32 = 1u;

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters controlling particle update for this frame.
struct ParticleUpdateParams {
    /// Number of particles to process.
    num_particles: u32,
    /// Delta time since last frame (seconds).
    delta_time: f32,
    /// Current simulation time (seconds).
    time: f32,
    /// Padding for 16-byte alignment.
    _padding: f32,
}

/// Force configuration for particle simulation.
/// All forces are applied additively to particle velocity.
struct ForceConfig {
    // -- Gravity (constant directional force) --
    /// Gravity direction (normalized).
    gravity: vec3<f32>,
    /// Gravity strength (m/s^2, typically 9.8).
    gravity_strength: f32,

    // -- Wind (constant directional force with noise) --
    /// Wind direction (normalized).
    wind: vec3<f32>,
    /// Wind strength (m/s^2).
    wind_strength: f32,

    // -- Turbulence (3D noise-based force) --
    /// Turbulence noise frequency (higher = more chaotic).
    turbulence_frequency: f32,
    /// Turbulence force amplitude.
    turbulence_amplitude: f32,
    /// Padding for alignment.
    _pad0: vec2<f32>,

    // -- Vortex (rotational force around axis) --
    /// Vortex center position (world space).
    vortex_center: vec3<f32>,
    /// Vortex rotation strength.
    vortex_strength: f32,

    /// Vortex rotation axis (normalized).
    vortex_axis: vec3<f32>,
    /// Vortex falloff radius (force decreases beyond this).
    vortex_radius: f32,

    // -- Attraction (point attractor/repeller) --
    /// Attraction point position (world space).
    attraction_point: vec3<f32>,
    /// Attraction strength (negative = repel).
    attraction_strength: f32,

    // -- Drag (velocity damping) --
    /// Drag coefficient (0-1, higher = more damping).
    drag: f32,
    /// Enable flags: bit 0=gravity, bit 1=wind, bit 2=turbulence, bit 3=vortex, bit 4=attraction.
    force_flags: u32,
    /// Padding for 16-byte alignment.
    _pad1: vec2<f32>,
}

/// GPU particle data (matches spawn.wgsl Particle struct, 64 bytes per particle).
/// Uses ping-pong buffer pattern for update stability.
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

/// Color interpolation parameters for lifetime-based color animation.
struct ColorParams {
    /// Starting color at age = 0.
    color_start: vec4<f32>,
    /// Ending color at age = lifetime.
    color_end: vec4<f32>,
    /// Starting size at age = 0.
    size_start: f32,
    /// Ending size at age = lifetime.
    size_end: f32,
    /// Padding for alignment.
    _pad: vec2<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: ParticleUpdateParams;
@group(0) @binding(1) var<uniform> forces: ForceConfig;
@group(0) @binding(2) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(3) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(4) var<storage, read_write> alive_flags: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> color_params: ColorParams;

// ============================================================================
// Simplex Noise Implementation (3D)
// ============================================================================
// Used for turbulence forces. Produces smooth, continuous noise.

/// Permutation table values (mod 12 for gradient indexing).
fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4((x * 34.0 + 1.0) * x);
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

/// 3D simplex noise returning value in [-1, 1].
fn simplex_noise_3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - D.yyy;

    // Permutations
    i = mod289_3(i);
    let p = permute(
        permute(
            permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        ) + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    // Gradients: 7x7 points over a square, mapped onto an octahedron
    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    // Normalize gradients
    let norm = taylor_inv_sqrt(vec4<f32>(
        dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)
    ));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    var m = max(0.6 - vec4<f32>(
        dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)
    ), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(
        dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)
    ));
}

/// 3D curl noise for divergence-free turbulence.
/// Returns a vector field that particles can follow without clumping.
fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.01;
    let dx = vec3<f32>(e, 0.0, 0.0);
    let dy = vec3<f32>(0.0, e, 0.0);
    let dz = vec3<f32>(0.0, 0.0, e);

    // Sample noise at offset positions
    let n_y_pz = simplex_noise_3d(p + dy);
    let n_y_mz = simplex_noise_3d(p - dy);
    let n_z_py = simplex_noise_3d(p + dz);
    let n_z_my = simplex_noise_3d(p - dz);
    let n_x_pz = simplex_noise_3d(p + dx);
    let n_x_mz = simplex_noise_3d(p - dx);
    let n_z_px = simplex_noise_3d(p + dz);
    let n_z_mx = simplex_noise_3d(p - dz);
    let n_x_py = simplex_noise_3d(p + dx);
    let n_x_my = simplex_noise_3d(p - dx);
    let n_y_px = simplex_noise_3d(p + dy);
    let n_y_mx = simplex_noise_3d(p - dy);

    // Curl = cross product of gradient
    let inv_2e = 1.0 / (2.0 * e);
    return vec3<f32>(
        (n_z_py - n_z_my - n_y_pz + n_y_mz) * inv_2e,
        (n_x_pz - n_x_mz - n_z_px + n_z_mx) * inv_2e,
        (n_y_px - n_y_mx - n_x_py + n_x_my) * inv_2e
    );
}

// ============================================================================
// Force Application Functions
// ============================================================================

/// Apply gravity force to velocity.
fn apply_gravity(velocity: vec3<f32>, gravity: vec3<f32>, strength: f32, dt: f32) -> vec3<f32> {
    return velocity + gravity * strength * dt;
}

/// Apply wind force to velocity with slight noise variation.
fn apply_wind(velocity: vec3<f32>, wind: vec3<f32>, strength: f32, position: vec3<f32>, time: f32, dt: f32) -> vec3<f32> {
    // Add slight variation based on position
    let noise = simplex_noise_3d(position * 0.1 + time * 0.5);
    let wind_strength = strength * (1.0 + noise * 0.3);
    return velocity + wind * wind_strength * dt;
}

/// Apply turbulence force using 3D curl noise.
fn apply_turbulence(velocity: vec3<f32>, position: vec3<f32>, time: f32, frequency: f32, amplitude: f32, dt: f32) -> vec3<f32> {
    let sample_pos = position * frequency + time * 0.2;
    let turbulence = curl_noise(sample_pos) * amplitude;
    return velocity + turbulence * dt;
}

/// Apply vortex (rotational) force around an axis.
fn apply_vortex(
    velocity: vec3<f32>,
    position: vec3<f32>,
    center: vec3<f32>,
    axis: vec3<f32>,
    strength: f32,
    radius: f32,
    dt: f32
) -> vec3<f32> {
    // Vector from center to particle
    let to_particle = position - center;

    // Project onto plane perpendicular to axis
    let along_axis = dot(to_particle, axis) * axis;
    let in_plane = to_particle - along_axis;
    let dist = length(in_plane);

    if dist < 0.001 {
        return velocity;
    }

    // Calculate tangential direction (perpendicular to radius in plane)
    let tangent = cross(axis, normalize(in_plane));

    // Force decreases with distance, capped at vortex_radius
    let falloff = 1.0 - saturate(dist / max(radius, 0.001));
    let force = tangent * strength * falloff;

    return velocity + force * dt;
}

/// Apply attraction/repulsion force toward a point.
fn apply_attraction(
    velocity: vec3<f32>,
    position: vec3<f32>,
    attraction_point: vec3<f32>,
    strength: f32,
    dt: f32
) -> vec3<f32> {
    let to_point = attraction_point - position;
    let dist = length(to_point);

    if dist < 0.001 {
        return velocity;
    }

    let dir = to_point / dist;
    // Force decreases with distance squared (inverse square law)
    let force = dir * strength / max(dist * dist, 0.01);

    return velocity + force * dt;
}

/// Apply drag to velocity (exponential decay).
fn apply_drag(velocity: vec3<f32>, drag: f32, dt: f32) -> vec3<f32> {
    // Exponential decay: v' = v * exp(-drag * dt)
    // Approximation for small dt: v' = v * (1 - drag * dt)
    let factor = max(1.0 - drag * dt, 0.0);
    return velocity * factor;
}

/// Apply all enabled forces to velocity.
fn apply_all_forces(
    particle: Particle,
    forces: ForceConfig,
    time: f32,
    dt: f32
) -> vec3<f32> {
    var velocity = particle.velocity;

    // Gravity (force_flags bit 0)
    if (forces.force_flags & 1u) != 0u {
        velocity = apply_gravity(velocity, forces.gravity, forces.gravity_strength, dt);
    }

    // Wind (force_flags bit 1)
    if (forces.force_flags & 2u) != 0u {
        velocity = apply_wind(velocity, forces.wind, forces.wind_strength, particle.position, time, dt);
    }

    // Turbulence (force_flags bit 2)
    if (forces.force_flags & 4u) != 0u {
        velocity = apply_turbulence(
            velocity,
            particle.position,
            time,
            forces.turbulence_frequency,
            forces.turbulence_amplitude,
            dt
        );
    }

    // Vortex (force_flags bit 3)
    if (forces.force_flags & 8u) != 0u {
        velocity = apply_vortex(
            velocity,
            particle.position,
            forces.vortex_center,
            forces.vortex_axis,
            forces.vortex_strength,
            forces.vortex_radius,
            dt
        );
    }

    // Attraction (force_flags bit 4)
    if (forces.force_flags & 16u) != 0u {
        velocity = apply_attraction(
            velocity,
            particle.position,
            forces.attraction_point,
            forces.attraction_strength,
            dt
        );
    }

    // Drag is always applied (not gated by flags)
    velocity = apply_drag(velocity, forces.drag, dt);

    return velocity;
}

// ============================================================================
// Color and Size Interpolation
// ============================================================================

/// Interpolate color based on age/lifetime ratio.
fn interpolate_color(age: f32, lifetime: f32, start: vec4<f32>, end: vec4<f32>) -> vec4<f32> {
    let t = saturate(age / max(lifetime, 0.001));
    return mix(start, end, t);
}

/// Interpolate size based on age/lifetime ratio.
fn interpolate_size(age: f32, lifetime: f32, start: f32, end: f32) -> f32 {
    let t = saturate(age / max(lifetime, 0.001));
    return mix(start, end, t);
}

// ============================================================================
// Main Entry Points
// ============================================================================

/// Main particle update kernel.
/// Reads from particles_in, writes to particles_out (ping-pong pattern).
@compute @workgroup_size(256)
fn update_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if idx >= params.num_particles {
        return;
    }

    // Read particle from input buffer
    var p = particles_in[idx];

    // Check if already dead
    if (p.flags & FLAG_ALIVE) == 0u {
        // Write dead particle unchanged
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    // Advance age
    p.age += params.delta_time;

    // Check if particle should die
    if p.age >= p.lifetime {
        // Mark as dead
        p.flags = p.flags & ~FLAG_ALIVE;
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    // Apply all forces to velocity
    p.velocity = apply_all_forces(p, forces, params.time, params.delta_time);

    // Integrate position
    p.position += p.velocity * params.delta_time;

    // Interpolate color over lifetime
    p.color = interpolate_color(p.age, p.lifetime, color_params.color_start, color_params.color_end);

    // Interpolate size over lifetime
    p.size = interpolate_size(p.age, p.lifetime, color_params.size_start, color_params.size_end);

    // Advance rotation
    p.rotation += p.rotation_speed * params.delta_time;
    // Wrap rotation to [0, 2*PI] for numerical stability
    p.rotation = p.rotation - floor(p.rotation / TWO_PI) * TWO_PI;

    // Write updated particle to output buffer
    particles_out[idx] = p;

    // Mark as alive in alive_flags for compaction pass
    atomicStore(&alive_flags[idx], 1u);
}

/// Variant: Update particles without color/size interpolation.
/// More efficient when using texture-based coloring.
@compute @workgroup_size(256)
fn update_particles_physics_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if idx >= params.num_particles {
        return;
    }

    var p = particles_in[idx];

    if (p.flags & FLAG_ALIVE) == 0u {
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    p.age += params.delta_time;

    if p.age >= p.lifetime {
        p.flags = p.flags & ~FLAG_ALIVE;
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    // Apply forces
    p.velocity = apply_all_forces(p, forces, params.time, params.delta_time);

    // Integrate position
    p.position += p.velocity * params.delta_time;

    // Advance rotation
    p.rotation += p.rotation_speed * params.delta_time;
    p.rotation = p.rotation - floor(p.rotation / TWO_PI) * TWO_PI;

    particles_out[idx] = p;
    atomicStore(&alive_flags[idx], 1u);
}

/// Variant: Simple gravity-only update (fastest path).
@compute @workgroup_size(256)
fn update_particles_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if idx >= params.num_particles {
        return;
    }

    var p = particles_in[idx];

    if (p.flags & FLAG_ALIVE) == 0u {
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    p.age += params.delta_time;

    if p.age >= p.lifetime {
        p.flags = p.flags & ~FLAG_ALIVE;
        particles_out[idx] = p;
        atomicStore(&alive_flags[idx], 0u);
        return;
    }

    // Simple gravity + drag only
    p.velocity = apply_gravity(p.velocity, forces.gravity, forces.gravity_strength, params.delta_time);
    p.velocity = apply_drag(p.velocity, forces.drag, params.delta_time);
    p.position += p.velocity * params.delta_time;

    particles_out[idx] = p;
    atomicStore(&alive_flags[idx], 1u);
}
