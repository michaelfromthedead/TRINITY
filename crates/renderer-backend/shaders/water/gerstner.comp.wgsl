// SPDX-License-Identifier: MIT
//
// gerstner.comp.wgsl - Gerstner Wave Compute Shader for TRINITY Engine (T-ENV-1.7)
//
// Evaluates Gerstner wave displacement for water surface vertices.
// Supports up to 32 superimposed waves for realistic ocean simulation.
//
// Algorithm:
// 1. Each thread processes one vertex in the water mesh grid
// 2. For each wave, compute phase and displacement
// 3. Sum horizontal and vertical displacements
// 4. Compute analytic surface normal from wave gradients
// 5. Write displaced position and normal to output buffer
//
// Physics:
// - Deep water dispersion: omega^2 = g * k
// - Phase velocity: c = sqrt(g / k)
// - Horizontal displacement: Q * A * D * cos(phase)
// - Vertical displacement: A * sin(phase)
//
// Performance:
// - Workgroup size 8x8 = 64 threads for optimal GPU occupancy
// - Memory-coalesced reads from wave parameter buffer
// - Single output write per thread

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_WAVES: u32 = 32u;
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const GRAVITY: f32 = 9.81;
const EPSILON: f32 = 0.000001;

// ============================================================================
// Data Structures
// ============================================================================

/// Single Gerstner wave parameters (32 bytes).
struct GerstnerWave {
    /// Wave amplitude (height from trough to crest / 2).
    amplitude: f32,
    /// Wavelength (horizontal distance between crests).
    wavelength: f32,
    /// Steepness factor (0-1, controls choppiness).
    steepness: f32,
    /// Phase velocity multiplier for animation speed.
    speed: f32,
    /// Normalized direction vector (XZ plane).
    direction: vec2<f32>,
    /// Padding for 16-byte alignment.
    _padding: vec2<f32>,
}

/// Wave evaluation parameters.
struct WaveParams {
    /// Number of active waves (1-32).
    wave_count: u32,
    /// Current animation time in seconds.
    time: f32,
    /// Padding for alignment.
    _padding: vec2<u32>,
}

/// Grid parameters for vertex generation.
struct GridParams {
    /// Number of vertices per side.
    grid_size: u32,
    /// Spacing between vertices in world units.
    spacing: f32,
    /// Grid origin X coordinate.
    origin_x: f32,
    /// Grid origin Z coordinate.
    origin_z: f32,
}

/// Output vertex data.
struct GerstnerVertex {
    /// Displaced world position.
    position: vec3<f32>,
    /// Padding for alignment.
    _pad0: f32,
    /// Surface normal.
    normal: vec3<f32>,
    /// Padding for alignment.
    _pad1: f32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Wave parameters uniform.
@group(0) @binding(0)
var<uniform> wave_params: WaveParams;

/// Grid parameters uniform.
@group(0) @binding(1)
var<uniform> grid_params: GridParams;

/// Array of wave configurations (up to 32 waves).
@group(0) @binding(2)
var<storage, read> waves: array<GerstnerWave, MAX_WAVES>;

/// Output vertex buffer (displaced positions + normals).
@group(0) @binding(3)
var<storage, read_write> output_vertices: array<GerstnerVertex>;

// ============================================================================
// Wave Physics Functions
// ============================================================================

/// Compute wave number k = 2*PI / wavelength.
fn wave_number(wavelength: f32) -> f32 {
    return TAU / max(wavelength, 0.1);
}

/// Compute angular frequency from deep water dispersion relation.
/// omega = sqrt(g * k)
fn angular_frequency(k: f32) -> f32 {
    return sqrt(GRAVITY * k);
}

/// Compute wave phase at position (x, z) and time t.
fn compute_phase(wave: GerstnerWave, x: f32, z: f32, time: f32) -> f32 {
    let k = wave_number(wave.wavelength);
    let omega = angular_frequency(k);
    let dot_product = wave.direction.x * x + wave.direction.y * z;
    return k * dot_product - omega * time * wave.speed;
}

/// Evaluate single wave displacement at (x, z).
fn evaluate_wave_displacement(wave: GerstnerWave, x: f32, z: f32, time: f32) -> vec3<f32> {
    let phase = compute_phase(wave, x, z, time);
    let cos_phase = cos(phase);
    let sin_phase = sin(phase);

    // Q factor for horizontal displacement
    let q = wave.steepness;

    // Displacement: horizontal follows direction * Q * A * cos, vertical = A * sin
    let dx = q * wave.amplitude * wave.direction.x * cos_phase;
    let dy = wave.amplitude * sin_phase;
    let dz = q * wave.amplitude * wave.direction.y * cos_phase;

    return vec3<f32>(dx, dy, dz);
}

/// Evaluate wave gradient for normal calculation.
/// Returns (dP/dx, dP/dz) as a mat2x3.
fn evaluate_wave_gradient(wave: GerstnerWave, x: f32, z: f32, time: f32) -> mat2x3<f32> {
    let phase = compute_phase(wave, x, z, time);
    let cos_phase = cos(phase);
    let sin_phase = sin(phase);

    let k = wave_number(wave.wavelength);
    let q = wave.steepness;
    let wa = k * wave.amplitude;

    // Partial derivatives
    // dP/dx
    let dx_dx = -q * wa * wave.direction.x * wave.direction.x * sin_phase;
    let dy_dx = wa * wave.direction.x * cos_phase;
    let dz_dx = -q * wa * wave.direction.x * wave.direction.y * sin_phase;

    // dP/dz
    let dx_dz = -q * wa * wave.direction.x * wave.direction.y * sin_phase;
    let dy_dz = wa * wave.direction.y * cos_phase;
    let dz_dz = -q * wa * wave.direction.y * wave.direction.y * sin_phase;

    return mat2x3<f32>(
        vec3<f32>(dx_dx, dy_dx, dz_dx),  // dP/dx
        vec3<f32>(dx_dz, dy_dz, dz_dz)   // dP/dz
    );
}

/// Compute surface normal from tangent vectors.
fn compute_normal(tangent_x: vec3<f32>, tangent_z: vec3<f32>) -> vec3<f32> {
    // Normal = tangent_z cross tangent_x (order for upward-facing normal)
    let normal = cross(tangent_z, tangent_x);

    let len = length(normal);
    if (len > EPSILON) {
        return normal / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    // Bounds check
    if (row >= grid_params.grid_size || col >= grid_params.grid_size) {
        return;
    }

    // Compute original grid position
    let x = grid_params.origin_x + f32(row) * grid_params.spacing;
    let z = grid_params.origin_z + f32(col) * grid_params.spacing;
    let time = wave_params.time;
    let num_waves = min(wave_params.wave_count, MAX_WAVES);

    // Accumulate displacement and gradients from all waves
    var total_displacement = vec3<f32>(0.0, 0.0, 0.0);
    var grad_x = vec3<f32>(0.0, 0.0, 0.0);
    var grad_z = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < num_waves; i = i + 1u) {
        let wave = waves[i];

        // Skip zero-amplitude waves
        if (wave.amplitude < EPSILON) {
            continue;
        }

        // Accumulate displacement
        total_displacement = total_displacement + evaluate_wave_displacement(wave, x, z, time);

        // Accumulate gradients
        let gradient = evaluate_wave_gradient(wave, x, z, time);
        grad_x = grad_x + gradient[0];
        grad_z = grad_z + gradient[1];
    }

    // Final displaced position
    let displaced_position = vec3<f32>(
        x + total_displacement.x,
        total_displacement.y,
        z + total_displacement.z
    );

    // Compute tangent vectors (identity + accumulated gradients)
    let tangent_x = vec3<f32>(1.0 + grad_x.x, grad_x.y, grad_x.z);
    let tangent_z = vec3<f32>(grad_z.x, grad_z.y, 1.0 + grad_z.z);

    // Compute surface normal
    let normal = compute_normal(tangent_x, tangent_z);

    // Write output
    let vertex_index = row * grid_params.grid_size + col;
    output_vertices[vertex_index].position = displaced_position;
    output_vertices[vertex_index].normal = normal;
}

// ============================================================================
// Single-Point Evaluation Entry Point
// ============================================================================

/// Alternative entry point for evaluating a single point (used for buoyancy queries).
/// Uses binding 4 for input positions and writes to same output buffer.
@compute
@workgroup_size(64, 1, 1)
fn evaluate_points(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = global_id.x;
    let total_points = num_workgroups.x * 64u;

    if (idx >= total_points) {
        return;
    }

    // Read input position (reusing output buffer for in-place transform)
    let input_pos = output_vertices[idx].position;
    let x = input_pos.x;
    let z = input_pos.z;
    let time = wave_params.time;
    let num_waves = min(wave_params.wave_count, MAX_WAVES);

    // Same accumulation logic as main entry point
    var total_displacement = vec3<f32>(0.0, 0.0, 0.0);
    var grad_x = vec3<f32>(0.0, 0.0, 0.0);
    var grad_z = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < num_waves; i = i + 1u) {
        let wave = waves[i];

        if (wave.amplitude < EPSILON) {
            continue;
        }

        total_displacement = total_displacement + evaluate_wave_displacement(wave, x, z, time);

        let gradient = evaluate_wave_gradient(wave, x, z, time);
        grad_x = grad_x + gradient[0];
        grad_z = grad_z + gradient[1];
    }

    let displaced_position = vec3<f32>(
        x + total_displacement.x,
        total_displacement.y,
        z + total_displacement.z
    );

    let tangent_x = vec3<f32>(1.0 + grad_x.x, grad_x.y, grad_x.z);
    let tangent_z = vec3<f32>(grad_z.x, grad_z.y, 1.0 + grad_z.z);
    let normal = compute_normal(tangent_x, tangent_z);

    output_vertices[idx].position = displaced_position;
    output_vertices[idx].normal = normal;
}
