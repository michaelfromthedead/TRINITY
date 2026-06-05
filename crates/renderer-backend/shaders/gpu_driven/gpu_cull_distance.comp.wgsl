// SPDX-License-Identifier: MIT
//
// gpu_cull_distance.comp.wgsl - Distance and LOD Culling for TRINITY Engine (T-GPU-3.2)
//
// Performs distance-based culling and LOD selection for GPU-driven rendering.
// One thread per instance, outputs LOD level or cull marker.
//
// Algorithm:
// 1. Compute distance from camera to instance center
// 2. If distance > max_draw_distance: cull (return 0xFFFFFFFF)
// 3. Otherwise: select LOD based on distance and lod_distances array
// 4. Apply LOD bias to shift quality up or down
// 5. Compute approximate screen size for streaming priority
//
// Performance: O(n) work, single dispatch, <0.05ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Maximum number of LOD levels per instance.
const MAX_LODS: u32 = 8u;

/// Marker value indicating instance is culled.
const CULLED_LOD: u32 = 0xFFFFFFFFu;

// ============================================================================
// Structs
// ============================================================================

/// Culling parameters uniform buffer.
///
/// Memory Layout (48 bytes, std140 aligned):
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | num_instances     | 4    |
/// | 4      | _pad0             | 4    |
/// | 8      | _pad1             | 4    |
/// | 12     | _pad2             | 4    |
/// | 16     | camera_position   | 12   |
/// | 28     | max_draw_distance | 4    |
/// | 32     | lod_bias          | 4    |
/// | 36     | _pad3             | 4    |
/// | 40     | _pad4             | 4    |
/// | 44     | _pad5             | 4    |
struct DistanceCullParams {
    /// Number of instances to process.
    num_instances: u32,
    /// Padding for vec4 alignment before camera_position.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// Camera position in world space.
    camera_position: vec3<f32>,
    /// Maximum draw distance (objects beyond are culled).
    max_draw_distance: f32,
    /// Global LOD bias: negative = higher quality, positive = lower quality.
    /// Range: typically -2.0 to +2.0
    lod_bias: f32,
    /// Padding for 16-byte alignment.
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

/// Per-instance LOD configuration.
///
/// Memory Layout (64 bytes):
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | position      | 12   |
/// | 12     | radius        | 4    |
/// | 16     | lod_distances | 32   |
/// | 48     | num_lods      | 4    |
/// | 52     | _pad          | 12   |
struct InstanceLOD {
    /// Instance position in world space (typically bounding sphere center).
    position: vec3<f32>,
    /// Bounding radius for screen size calculation.
    radius: f32,
    /// LOD transition distances: lod_distances[i] = max distance for LOD level i.
    /// If distance < lod_distances[0], use LOD 0 (highest quality).
    /// If lod_distances[i-1] <= distance < lod_distances[i], use LOD i.
    lod_distances: array<f32, 8>,
    /// Number of LOD levels for this instance (1-8).
    num_lods: u32,
    /// Padding for 16-byte alignment.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Output LOD result for each instance.
///
/// Memory Layout (8 bytes):
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | lod_level   | 4    |
/// | 4      | screen_size | 4    |
struct LODResult {
    /// Selected LOD level (0 = highest quality, or CULLED_LOD = 0xFFFFFFFF for culled).
    lod_level: u32,
    /// Approximate projected screen size in normalized units (0-1).
    /// Used for streaming priority: larger values = more important.
    screen_size: f32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Culling parameters (uniform buffer).
@group(0) @binding(0) var<uniform> params: DistanceCullParams;

/// Instance LOD configurations (read-only storage buffer).
@group(0) @binding(1) var<storage, read> instances: array<InstanceLOD>;

/// Output LOD results (read-write storage buffer).
@group(0) @binding(2) var<storage, read_write> results: array<LODResult>;

// ============================================================================
// LOD Selection
// ============================================================================

/// Select LOD level based on distance and LOD transition thresholds.
///
/// The LOD bias shifts the effective distance:
/// - Negative bias = treat objects as closer = higher quality LOD
/// - Positive bias = treat objects as farther = lower quality LOD
///
/// LOD selection logic:
/// - If biased_distance < lod_distances[0]: LOD 0 (highest quality)
/// - If lod_distances[i-1] <= biased_distance < lod_distances[i]: LOD i
/// - If biased_distance >= lod_distances[num_lods-1]: LOD num_lods-1 (lowest quality)
fn select_lod(distance: f32, instance: InstanceLOD, bias: f32) -> u32 {
    // Apply LOD bias: bias shifts the effective distance.
    // pow(2, bias) means bias of +1 doubles distance, -1 halves it.
    let biased_distance = distance * pow(2.0, bias);

    // Find the first LOD level where biased_distance < threshold.
    for (var lod = 0u; lod < instance.num_lods; lod++) {
        if (biased_distance < instance.lod_distances[lod]) {
            return lod;
        }
    }

    // Beyond all thresholds: use lowest quality LOD (still render, not culled).
    return instance.num_lods - 1u;
}

// ============================================================================
// Screen Size Estimation
// ============================================================================

/// Compute approximate screen size based on distance and bounding radius.
///
/// Returns size in normalized device coordinates (0-1 range).
/// This is a rough approximation used for streaming priority, not for
/// pixel-accurate culling. Assumes a reasonable FOV.
///
/// Formula: screen_size ~ radius / distance
/// At distance = radius, object fills ~1 unit (very close).
/// At distance = 10 * radius, object fills ~0.1 units.
fn compute_screen_size(distance: f32, radius: f32) -> f32 {
    // Guard against division by zero or very small distances.
    if (distance <= 0.001) {
        return 1.0;  // Very close = assume full screen.
    }

    // Simple approximation: angular size ~ radius / distance.
    // Clamped to [0, 1] for normalized output.
    return clamp(radius / distance, 0.0, 1.0);
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-instance distance culling and LOD selection kernel.
///
/// Each thread processes one instance:
/// 1. Compute distance from camera to instance center
/// 2. If beyond max_draw_distance: mark as culled
/// 3. Otherwise: select LOD and compute screen size
@compute @workgroup_size(256)
fn cull_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check: skip threads beyond instance count.
    if (idx >= params.num_instances) {
        return;
    }

    let instance = instances[idx];

    // Compute distance from camera to instance.
    let to_instance = instance.position - params.camera_position;
    let distance = length(to_instance);

    // Distance cull check: if beyond max draw distance, cull.
    if (distance > params.max_draw_distance) {
        results[idx] = LODResult(CULLED_LOD, 0.0);
        return;
    }

    // Select LOD based on distance and bias.
    let lod = select_lod(distance, instance, params.lod_bias);

    // Compute screen size for streaming priority.
    let screen_size = compute_screen_size(distance, instance.radius);

    results[idx] = LODResult(lod, screen_size);
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Distance-only culling (no LOD selection).
/// Marks instances as visible (0) or culled (CULLED_LOD).
/// Useful when all instances have the same mesh.
@compute @workgroup_size(256)
fn cull_distance_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let instance = instances[idx];
    let to_instance = instance.position - params.camera_position;
    let distance = length(to_instance);

    if (distance > params.max_draw_distance) {
        results[idx] = LODResult(CULLED_LOD, 0.0);
    } else {
        // Visible: LOD 0, with screen size.
        let screen_size = compute_screen_size(distance, instance.radius);
        results[idx] = LODResult(0u, screen_size);
    }
}

/// LOD selection only (no distance culling).
/// Always selects a LOD, never culls.
/// Useful when distance culling is handled by a separate pass.
@compute @workgroup_size(256)
fn lod_select_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let instance = instances[idx];
    let to_instance = instance.position - params.camera_position;
    let distance = length(to_instance);

    let lod = select_lod(distance, instance, params.lod_bias);
    let screen_size = compute_screen_size(distance, instance.radius);

    results[idx] = LODResult(lod, screen_size);
}
