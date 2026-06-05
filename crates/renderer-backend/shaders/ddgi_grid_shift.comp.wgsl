// SPDX-License-Identifier: MIT
//
// ddgi_grid_shift.comp.wgsl -- DDGI infinite scrolling grid shift compute shader.
//
// When the camera moves, probes at the trailing edge scroll out and need to be
// re-seeded with interpolated data from their new neighbors at the leading edge.
//
// This shader computes which probes need re-seeding based on scroll delta and
// initializes them with weighted averages from valid neighbors plus noise dither
// for temporal stability.
//
// Entry points:
//   1. ddgi_grid_shift -- Main grid shift kernel
//   2. ddgi_detect_stale_probes -- Mark probes needing re-seed (optional pass)

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const GOLDEN_RATIO: f32 = 1.6180339887498948482;

// Noise dither strength for newly seeded probes (prevents temporal aliasing)
const SEED_NOISE_STRENGTH: f32 = 0.05;

// Minimum neighbor weight to consider valid
const MIN_NEIGHBOR_WEIGHT: f32 = 0.001;

// Maximum distance for neighbor contribution (in grid cells)
const MAX_NEIGHBOR_DISTANCE: f32 = 2.0;

// Hysteresis for blending re-seeded probes
const RESEED_HYSTERESIS: f32 = 0.7;

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters for grid shift operation.
struct GridShiftParams {
    /// Previous scroll offset (grid cell units)
    old_scroll_offset: vec3<i32>,
    /// Padding for alignment
    _pad0: i32,
    /// New scroll offset (grid cell units)
    new_scroll_offset: vec3<i32>,
    /// Padding for alignment
    _pad1: i32,
    /// Grid dimensions (probes per axis)
    dimensions: vec3<u32>,
    /// Padding for alignment
    _pad2: u32,
    /// Frame index for noise dithering
    frame_index: u32,
    /// Grid cell spacing in world units
    cell_spacing: f32,
    /// Seed blend factor (0.0 = keep old, 1.0 = full reseed)
    seed_blend_factor: f32,
    /// Reserved
    _reserved: f32,
}

/// Per-probe spherical harmonics storage (L2, 9 coefficients RGB).
/// Matches Rust ProbeSH struct layout: 192 bytes.
struct ProbeSH {
    /// 9 RGB coefficients as vec4 (w unused)
    irradiance: array<vec4<f32>, 9>,
    /// 9 visibility coefficients
    visibility: array<f32, 9>,
    /// Padding to 192 bytes
    _pad: array<f32, 3>,
}

/// Probe status flags for re-seeding
struct ProbeStatus {
    /// Needs re-seeding this frame
    needs_reseed: u32,
    /// Confidence level (0 = stale, 255 = fully converged)
    confidence: u32,
    /// Frames since last update
    stale_frames: u32,
    /// Reserved
    _reserved: u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: GridShiftParams;
@group(0) @binding(1) var<storage, read_write> probes: array<ProbeSH>;
@group(0) @binding(2) var<storage, read_write> probe_status: array<ProbeStatus>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute linear buffer index from 3D grid index.
fn grid_to_linear(idx: vec3<u32>) -> u32 {
    return idx.x + idx.y * params.dimensions.x + idx.z * params.dimensions.x * params.dimensions.y;
}

/// Compute 3D grid index from linear buffer index.
fn linear_to_grid(linear: u32) -> vec3<u32> {
    let xy_size = params.dimensions.x * params.dimensions.y;
    let z = linear / xy_size;
    let remainder = linear % xy_size;
    let y = remainder / params.dimensions.x;
    let x = remainder % params.dimensions.x;
    return vec3<u32>(x, y, z);
}

/// Apply scroll offset with wrapping.
fn apply_scroll(idx: vec3<u32>, offset: vec3<i32>) -> vec3<u32> {
    let sx = u32((i32(idx.x) + offset.x) % i32(params.dimensions.x));
    let sy = u32((i32(idx.y) + offset.y) % i32(params.dimensions.y));
    let sz = u32((i32(idx.z) + offset.z) % i32(params.dimensions.z));
    return vec3<u32>(
        select(sx, sx + params.dimensions.x, i32(sx) < 0),
        select(sy, sy + params.dimensions.y, i32(sy) < 0),
        select(sz, sz + params.dimensions.z, i32(sz) < 0)
    );
}

/// Euclidean modulo (always positive result).
fn emod(a: i32, b: u32) -> u32 {
    let m = a % i32(b);
    return u32(select(m + i32(b), m, m >= 0));
}

/// Apply scroll offset with proper euclidean modulo.
fn apply_scroll_emod(idx: vec3<u32>, offset: vec3<i32>) -> vec3<u32> {
    return vec3<u32>(
        emod(i32(idx.x) + offset.x, params.dimensions.x),
        emod(i32(idx.y) + offset.y, params.dimensions.y),
        emod(i32(idx.z) + offset.z, params.dimensions.z)
    );
}

/// Check if a probe needs re-seeding based on scroll delta.
/// Returns true if the probe scrolled in from the boundary.
fn probe_needs_reseed(grid_idx: vec3<u32>, delta: vec3<i32>) -> bool {
    // A probe needs re-seeding if it's in the "new" region that scrolled in.
    // This region is defined by probes whose wrapped position changed significantly.

    // For each axis, check if this probe is in the scroll-in region
    var needs_reseed = false;

    // X axis scroll
    if delta.x > 0 {
        // Scrolling positive: new probes at high x indices
        let threshold_x = params.dimensions.x - u32(delta.x);
        needs_reseed = needs_reseed || (grid_idx.x >= threshold_x);
    } else if delta.x < 0 {
        // Scrolling negative: new probes at low x indices
        needs_reseed = needs_reseed || (grid_idx.x < u32(-delta.x));
    }

    // Y axis scroll
    if delta.y > 0 {
        let threshold_y = params.dimensions.y - u32(delta.y);
        needs_reseed = needs_reseed || (grid_idx.y >= threshold_y);
    } else if delta.y < 0 {
        needs_reseed = needs_reseed || (grid_idx.y < u32(-delta.y));
    }

    // Z axis scroll
    if delta.z > 0 {
        let threshold_z = params.dimensions.z - u32(delta.z);
        needs_reseed = needs_reseed || (grid_idx.z >= threshold_z);
    } else if delta.z < 0 {
        needs_reseed = needs_reseed || (grid_idx.z < u32(-delta.z));
    }

    return needs_reseed;
}

/// Hash function for noise generation.
fn hash_vec3(v: vec3<u32>) -> u32 {
    var h = v.x * 374761393u + v.y * 668265263u + v.z * 2147483647u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

/// Generate pseudo-random value [0, 1) from probe index and frame.
fn noise_value(probe_idx: vec3<u32>, frame: u32) -> f32 {
    let h = hash_vec3(vec3<u32>(probe_idx.x, probe_idx.y, probe_idx.z + frame * 7919u));
    return f32(h) / 4294967296.0;
}

/// Generate 3D noise vector for dithering.
fn noise_vec3(probe_idx: vec3<u32>, frame: u32) -> vec3<f32> {
    return vec3<f32>(
        noise_value(probe_idx, frame) * 2.0 - 1.0,
        noise_value(probe_idx, frame + 1u) * 2.0 - 1.0,
        noise_value(probe_idx, frame + 2u) * 2.0 - 1.0
    );
}

/// Check if a grid index is valid (within bounds).
fn is_valid_index(idx: vec3<i32>) -> bool {
    return idx.x >= 0 && idx.x < i32(params.dimensions.x) &&
           idx.y >= 0 && idx.y < i32(params.dimensions.y) &&
           idx.z >= 0 && idx.z < i32(params.dimensions.z);
}

/// Compute distance-based falloff weight.
fn distance_weight(dist: f32) -> f32 {
    let normalized = dist / MAX_NEIGHBOR_DISTANCE;
    return max(0.0, 1.0 - normalized * normalized);
}

/// Lerp between two ProbeSH structs.
fn lerp_probe_sh(a: ProbeSH, b: ProbeSH, t: f32) -> ProbeSH {
    var result: ProbeSH;
    let inv_t = 1.0 - t;

    for (var i = 0u; i < 9u; i = i + 1u) {
        result.irradiance[i] = a.irradiance[i] * inv_t + b.irradiance[i] * t;
        result.visibility[i] = a.visibility[i] * inv_t + b.visibility[i] * t;
    }

    result._pad[0] = 0.0;
    result._pad[1] = 0.0;
    result._pad[2] = 0.0;

    return result;
}

/// Add two ProbeSH structs (weighted).
fn add_probe_sh_weighted(accum: ptr<function, ProbeSH>, other: ProbeSH, weight: f32) {
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*accum).irradiance[i] = (*accum).irradiance[i] + other.irradiance[i] * weight;
        (*accum).visibility[i] = (*accum).visibility[i] + other.visibility[i] * weight;
    }
}

/// Scale ProbeSH by a factor.
fn scale_probe_sh(probe: ptr<function, ProbeSH>, scale: f32) {
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*probe).irradiance[i] = (*probe).irradiance[i] * scale;
        (*probe).visibility[i] = (*probe).visibility[i] * scale;
    }
}

/// Create zero-initialized ProbeSH.
fn zero_probe_sh() -> ProbeSH {
    var result: ProbeSH;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.irradiance[i] = vec4<f32>(0.0);
        result.visibility[i] = 0.0;
    }
    result._pad[0] = 0.0;
    result._pad[1] = 0.0;
    result._pad[2] = 0.0;
    return result;
}

/// Apply noise dither to probe data (prevents temporal aliasing).
fn apply_noise_dither(probe: ptr<function, ProbeSH>, probe_idx: vec3<u32>, frame: u32) {
    let noise = noise_vec3(probe_idx, frame);
    let strength = SEED_NOISE_STRENGTH;

    // Dither L1 coefficients (directional term) - most visible
    for (var i = 1u; i < 4u; i = i + 1u) {
        (*probe).irradiance[i].x = (*probe).irradiance[i].x + noise.x * strength;
        (*probe).irradiance[i].y = (*probe).irradiance[i].y + noise.y * strength;
        (*probe).irradiance[i].z = (*probe).irradiance[i].z + noise.z * strength;
    }
}

// ============================================================================
// Neighbor Seeding
// ============================================================================

/// Seed a probe from weighted average of valid neighbors.
/// Uses distance-weighted blending with falloff.
fn seed_probe_from_neighbors(probe_idx: vec3<u32>, delta: vec3<i32>) -> ProbeSH {
    var accum = zero_probe_sh();
    var total_weight = 0.0;

    // Sample from a 3x3x3 neighborhood (excluding self and invalid neighbors)
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                // Skip self
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }

                let neighbor_idx = vec3<i32>(probe_idx) + vec3<i32>(dx, dy, dz);

                // Check bounds
                if !is_valid_index(neighbor_idx) {
                    continue;
                }

                let neighbor_u = vec3<u32>(neighbor_idx);

                // Skip neighbors that also need re-seeding (they're stale too)
                if probe_needs_reseed(neighbor_u, delta) {
                    continue;
                }

                // Compute distance weight
                let dist = sqrt(f32(dx * dx + dy * dy + dz * dz));
                let weight = distance_weight(dist);

                if weight < MIN_NEIGHBOR_WEIGHT {
                    continue;
                }

                // Accumulate weighted neighbor data
                let linear = grid_to_linear(neighbor_u);
                let neighbor = probes[linear];
                add_probe_sh_weighted(&accum, neighbor, weight);
                total_weight = total_weight + weight;
            }
        }
    }

    // Normalize by total weight
    if total_weight > MIN_NEIGHBOR_WEIGHT {
        scale_probe_sh(&accum, 1.0 / total_weight);
    } else {
        // No valid neighbors - initialize with minimal ambient
        accum.irradiance[0] = vec4<f32>(0.01, 0.01, 0.01, 0.0); // Tiny ambient
        for (var i = 0u; i < 9u; i = i + 1u) {
            accum.visibility[i] = 1.0; // Full visibility (conservative)
        }
    }

    return accum;
}

// ============================================================================
// Entry Point: Grid Shift
// ============================================================================

/// Main grid shift kernel.
///
/// For each probe, determines if it scrolled into a new position and needs
/// re-seeding. If so, computes weighted average from valid neighbors and
/// applies noise dither for temporal stability.
///
/// Dispatched as: ceil(total_probes / 64) workgroups of 64 threads.
@compute @workgroup_size(8, 8, 1)
fn ddgi_grid_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    let iz = gid.z;

    // Bounds check
    if ix >= params.dimensions.x || iy >= params.dimensions.y || iz >= params.dimensions.z {
        return;
    }

    let probe_idx = vec3<u32>(ix, iy, iz);
    let linear = grid_to_linear(probe_idx);

    // Compute scroll delta
    let delta = params.new_scroll_offset - params.old_scroll_offset;

    // Check if any scrolling happened
    if delta.x == 0 && delta.y == 0 && delta.z == 0 {
        return;
    }

    // Check if this probe needs re-seeding
    if !probe_needs_reseed(probe_idx, delta) {
        return;
    }

    // Get current probe data
    let old_probe = probes[linear];

    // Seed from neighbors
    var new_probe = seed_probe_from_neighbors(probe_idx, delta);

    // Apply noise dither for temporal stability
    apply_noise_dither(&new_probe, probe_idx, params.frame_index);

    // Blend with old data using hysteresis (helps if scroll is temporary)
    let blended = lerp_probe_sh(old_probe, new_probe, params.seed_blend_factor);

    // Write back
    probes[linear] = blended;

    // Update status
    probe_status[linear].needs_reseed = 0u;
    probe_status[linear].confidence = 32u; // Low confidence, needs updates
    probe_status[linear].stale_frames = 0u;
}

// ============================================================================
// Entry Point: Detect Stale Probes (Optional Pass)
// ============================================================================

/// Optional pre-pass to mark probes that need re-seeding.
/// Can be used for debugging or alternative scheduling.
@compute @workgroup_size(8, 8, 1)
fn ddgi_detect_stale_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    let iz = gid.z;

    // Bounds check
    if ix >= params.dimensions.x || iy >= params.dimensions.y || iz >= params.dimensions.z {
        return;
    }

    let probe_idx = vec3<u32>(ix, iy, iz);
    let linear = grid_to_linear(probe_idx);

    // Compute scroll delta
    let delta = params.new_scroll_offset - params.old_scroll_offset;

    // Mark probes that need re-seeding
    if probe_needs_reseed(probe_idx, delta) {
        probe_status[linear].needs_reseed = 1u;
        probe_status[linear].confidence = 0u;
    }
}
