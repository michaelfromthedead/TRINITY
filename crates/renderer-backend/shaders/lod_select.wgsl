// SPDX-License-Identifier: MIT
//
// lod_select.wgsl - LOD Selection Compute Shader (T-WGPU-P6.5.2)
//
// GPU-based Level of Detail (LOD) selection for objects. Supports both
// distance-based and screen-size-based LOD selection with optional
// blend factors for smooth LOD transitions.
//
// Memory Layout:
//   LodSelectParams: 48 bytes (camera info + LOD config)
//   ObjectLodInput:  48 bytes (position, radius, thresholds)
//   LodSelectOutput: 8 bytes  (level + blend_factor)
//
// Performance:
//   - Workgroup size: 64 threads (optimal GPU occupancy)
//   - One thread per object
//   - O(1) per object, O(n) total
//   - Minimal memory bandwidth (read 48B, write 8B per object)

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size (64 threads for optimal GPU occupancy)
const WORKGROUP_SIZE: u32 = 64u;

/// Number of LOD levels supported (0 = highest detail, 3 = lowest)
const NUM_LOD_LEVELS: u32 = 4u;

/// Small epsilon for floating point comparisons
const EPSILON: f32 = 1e-6;

/// LOD selection mode flags
const MODE_DISTANCE: u32 = 0u;      // Distance-based LOD selection
const MODE_SCREEN_SIZE: u32 = 1u;   // Screen-size-based LOD selection

/// Default screen coverage thresholds (matches Rust LOD constants)
const COVERAGE_LOD0: f32 = 0.10;  // >= 10% screen: LOD 0
const COVERAGE_LOD1: f32 = 0.04;  // >= 4% screen:  LOD 1
const COVERAGE_LOD2: f32 = 0.01;  // >= 1% screen:  LOD 2
                                   // < 1% screen:   LOD 3

// ============================================================================
// Structs
// ============================================================================

/// LOD selection parameters (camera and configuration).
///
/// Memory layout: 48 bytes, aligned to 16 bytes.
///
/// | Offset | Field           | Size | Description                      |
/// |--------|-----------------|------|----------------------------------|
/// | 0      | camera_position | 12   | Camera world position            |
/// | 12     | _pad0           | 4    | Padding for vec4 alignment       |
/// | 16     | screen_width    | 4    | Screen width in pixels           |
/// | 20     | screen_height   | 4    | Screen height in pixels          |
/// | 24     | fov_y           | 4    | Vertical FOV in radians          |
/// | 28     | selection_mode  | 4    | 0=distance, 1=screen-size        |
/// | 32     | object_count    | 4    | Number of objects to process     |
/// | 36     | enable_blend    | 4    | 1=calc blend factor, 0=skip      |
/// | 40     | blend_range     | 4    | Blend distance (% of threshold)  |
/// | 44     | _pad1           | 4    | Padding for 16-byte alignment    |
struct LodSelectParams {
    camera_position: vec3<f32>,
    _pad0: f32,
    screen_width: f32,
    screen_height: f32,
    fov_y: f32,
    selection_mode: u32,
    object_count: u32,
    enable_blend: u32,
    blend_range: f32,
    _pad1: f32,
}

/// Per-object LOD input data.
///
/// Memory layout: 48 bytes, aligned to 16 bytes.
///
/// | Offset | Field          | Size | Description                       |
/// |--------|----------------|------|-----------------------------------|
/// | 0      | world_position | 12   | Object center in world space      |
/// | 12     | bounding_radius| 4    | Bounding sphere radius            |
/// | 16     | thresholds     | 12   | LOD 0->1, 1->2, 2->3 distances    |
/// | 28     | _pad0          | 4    | Padding                           |
/// | 32     | flags          | 4    | Object flags (force LOD, etc.)    |
/// | 36     | forced_lod     | 4    | Forced LOD level (if flag set)    |
/// | 40     | _pad1          | 8    | Padding for 16-byte alignment     |
struct ObjectLodInput {
    world_position: vec3<f32>,
    bounding_radius: f32,
    thresholds: vec3<f32>,
    _pad0: f32,
    flags: u32,
    forced_lod: u32,
    _pad1: vec2<f32>,
}

/// LOD selection output per object.
///
/// Memory layout: 8 bytes.
///
/// | Offset | Field        | Size | Description                        |
/// |--------|--------------|------|------------------------------------|
/// | 0      | level        | 4    | Selected LOD level (0-3)           |
/// | 4      | blend_factor | 4    | Transition blend factor (0.0-1.0)  |
struct LodSelectOutput {
    level: u32,
    blend_factor: f32,
}

/// Object flags bitfield
const FLAG_FORCE_LOD: u32 = 1u;       // Use forced_lod instead of calculating
const FLAG_ALWAYS_LOD0: u32 = 2u;     // Always use highest detail
const FLAG_ALWAYS_LOD3: u32 = 4u;     // Always use lowest detail
const FLAG_DISABLE_BLEND: u32 = 8u;   // Disable blend factor for this object

// ============================================================================
// Bindings
// ============================================================================

/// Bind group 0: LOD parameters (uniform, read-only)
@group(0) @binding(0) var<uniform> params: LodSelectParams;

/// Bind group 1: Object LOD input data (storage, read-only)
@group(1) @binding(0) var<storage, read> objects: array<ObjectLodInput>;

/// Bind group 1: LOD selection output (storage, read-write)
@group(1) @binding(1) var<storage, read_write> output: array<LodSelectOutput>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate distance from camera to object center.
fn distance_to_camera(object_pos: vec3<f32>) -> f32 {
    let diff = object_pos - params.camera_position;
    return length(diff);
}

/// Calculate squared distance (avoids sqrt for threshold comparison).
fn distance_to_camera_squared(object_pos: vec3<f32>) -> f32 {
    let diff = object_pos - params.camera_position;
    return dot(diff, diff);
}

/// Calculate screen coverage for an object.
///
/// Returns the fraction of screen height covered by the object's bounding sphere.
/// Coverage > 1.0 means the object extends beyond the screen.
fn screen_coverage(object_pos: vec3<f32>, radius: f32) -> f32 {
    let distance = distance_to_camera(object_pos);

    // Handle edge case: object at camera position
    if distance < EPSILON {
        return 1.0;
    }

    let half_fov = params.fov_y * 0.5;
    let tan_half_fov = tan(half_fov);

    // Handle degenerate FOV
    if tan_half_fov < EPSILON {
        return 1.0;
    }

    // Visible height at object's distance
    let visible_height = 2.0 * distance * tan_half_fov;

    // Object diameter / visible height = coverage
    let diameter = 2.0 * radius;
    return diameter / visible_height;
}

/// Select LOD level based on distance.
///
/// Compares distance against thresholds to select LOD 0-3.
fn select_lod_by_distance(distance: f32, thresholds: vec3<f32>) -> u32 {
    if distance < thresholds.x {
        return 0u;  // Closest: highest detail
    } else if distance < thresholds.y {
        return 1u;
    } else if distance < thresholds.z {
        return 2u;
    } else {
        return 3u;  // Farthest: lowest detail
    }
}

/// Select LOD level based on screen coverage.
///
/// Uses standard coverage thresholds (10%, 4%, 1%).
fn select_lod_by_coverage(coverage: f32) -> u32 {
    if coverage >= COVERAGE_LOD0 {
        return 0u;  // Large on screen: highest detail
    } else if coverage >= COVERAGE_LOD1 {
        return 1u;
    } else if coverage >= COVERAGE_LOD2 {
        return 2u;
    } else {
        return 3u;  // Tiny on screen: lowest detail
    }
}

/// Calculate blend factor for smooth LOD transitions.
///
/// Returns 0.0 when fully at current LOD, 1.0 when at transition boundary.
/// The blend_range parameter controls how wide the blend zone is
/// (as a fraction of the distance to next threshold).
fn calculate_blend_factor(distance: f32, thresholds: vec3<f32>, current_lod: u32) -> f32 {
    // Get the relevant threshold for current LOD
    var threshold: f32;
    var prev_threshold: f32;

    if current_lod == 0u {
        threshold = thresholds.x;
        prev_threshold = 0.0;
    } else if current_lod == 1u {
        threshold = thresholds.y;
        prev_threshold = thresholds.x;
    } else if current_lod == 2u {
        threshold = thresholds.z;
        prev_threshold = thresholds.y;
    } else {
        // LOD 3 has no transition (it's the lowest)
        return 0.0;
    }

    // Calculate blend zone start (blend_range before threshold)
    let range = threshold - prev_threshold;
    let blend_start = threshold - (range * params.blend_range);

    // Calculate blend factor (0.0 at blend_start, 1.0 at threshold)
    if distance < blend_start {
        return 0.0;
    } else if distance >= threshold {
        return 1.0;
    } else {
        return (distance - blend_start) / (threshold - blend_start);
    }
}

/// Calculate blend factor for screen-size based LOD.
fn calculate_blend_factor_coverage(coverage: f32, current_lod: u32) -> f32 {
    // Coverage thresholds (higher = closer to LOD 0)
    var threshold: f32;
    var next_threshold: f32;

    if current_lod == 0u {
        threshold = COVERAGE_LOD0;
        next_threshold = COVERAGE_LOD1;
    } else if current_lod == 1u {
        threshold = COVERAGE_LOD1;
        next_threshold = COVERAGE_LOD2;
    } else if current_lod == 2u {
        threshold = COVERAGE_LOD2;
        next_threshold = 0.0;
    } else {
        return 0.0;
    }

    // Blend zone at lower end of coverage range
    let range = threshold - next_threshold;
    let blend_start = threshold - (range * params.blend_range);

    if coverage >= blend_start {
        return 0.0;  // Solidly in current LOD
    } else if coverage <= next_threshold {
        return 1.0;  // Transitioned to next LOD
    } else {
        return (blend_start - coverage) / (blend_start - next_threshold);
    }
}

// ============================================================================
// Main Compute Entry Point
// ============================================================================

/// LOD selection compute shader main entry point.
///
/// Each thread processes one object:
/// 1. Check for forced LOD flags
/// 2. Calculate distance or screen coverage
/// 3. Select appropriate LOD level (0-3)
/// 4. Optionally calculate blend factor for smooth transitions
@compute @workgroup_size(64, 1, 1)
fn lod_select_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    // Bounds check
    if object_idx >= params.object_count {
        return;
    }

    // Read object data
    let obj = objects[object_idx];

    // Initialize output
    var result: LodSelectOutput;
    result.level = 0u;
    result.blend_factor = 0.0;

    // Check for forced LOD flags
    if (obj.flags & FLAG_FORCE_LOD) != 0u {
        result.level = min(obj.forced_lod, NUM_LOD_LEVELS - 1u);
        output[object_idx] = result;
        return;
    }

    if (obj.flags & FLAG_ALWAYS_LOD0) != 0u {
        result.level = 0u;
        output[object_idx] = result;
        return;
    }

    if (obj.flags & FLAG_ALWAYS_LOD3) != 0u {
        result.level = 3u;
        output[object_idx] = result;
        return;
    }

    // Select LOD based on mode
    if params.selection_mode == MODE_SCREEN_SIZE {
        // Screen-size based LOD selection
        let coverage = screen_coverage(obj.world_position, obj.bounding_radius);
        result.level = select_lod_by_coverage(coverage);

        // Calculate blend factor if enabled
        if params.enable_blend != 0u && (obj.flags & FLAG_DISABLE_BLEND) == 0u {
            result.blend_factor = calculate_blend_factor_coverage(coverage, result.level);
        }
    } else {
        // Distance-based LOD selection (default)
        let distance = distance_to_camera(obj.world_position);
        result.level = select_lod_by_distance(distance, obj.thresholds);

        // Calculate blend factor if enabled
        if params.enable_blend != 0u && (obj.flags & FLAG_DISABLE_BLEND) == 0u {
            result.blend_factor = calculate_blend_factor(distance, obj.thresholds, result.level);
        }
    }

    // Write result
    output[object_idx] = result;
}

// ============================================================================
// Variant Entry Points
// ============================================================================

/// Distance-only LOD selection (no blend factor calculation).
///
/// Optimized path when blend factors are not needed.
@compute @workgroup_size(64, 1, 1)
fn lod_select_distance_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    if object_idx >= params.object_count {
        return;
    }

    let obj = objects[object_idx];

    // Handle forced LOD
    if (obj.flags & FLAG_FORCE_LOD) != 0u {
        output[object_idx] = LodSelectOutput(min(obj.forced_lod, 3u), 0.0);
        return;
    }
    if (obj.flags & FLAG_ALWAYS_LOD0) != 0u {
        output[object_idx] = LodSelectOutput(0u, 0.0);
        return;
    }
    if (obj.flags & FLAG_ALWAYS_LOD3) != 0u {
        output[object_idx] = LodSelectOutput(3u, 0.0);
        return;
    }

    let distance = distance_to_camera(obj.world_position);
    let level = select_lod_by_distance(distance, obj.thresholds);

    output[object_idx] = LodSelectOutput(level, 0.0);
}

/// Screen-size-only LOD selection (no blend factor calculation).
///
/// Optimized path for screen-size LOD without blending.
@compute @workgroup_size(64, 1, 1)
fn lod_select_screen_size_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    if object_idx >= params.object_count {
        return;
    }

    let obj = objects[object_idx];

    // Handle forced LOD
    if (obj.flags & FLAG_FORCE_LOD) != 0u {
        output[object_idx] = LodSelectOutput(min(obj.forced_lod, 3u), 0.0);
        return;
    }
    if (obj.flags & FLAG_ALWAYS_LOD0) != 0u {
        output[object_idx] = LodSelectOutput(0u, 0.0);
        return;
    }
    if (obj.flags & FLAG_ALWAYS_LOD3) != 0u {
        output[object_idx] = LodSelectOutput(3u, 0.0);
        return;
    }

    let coverage = screen_coverage(obj.world_position, obj.bounding_radius);
    let level = select_lod_by_coverage(coverage);

    output[object_idx] = LodSelectOutput(level, 0.0);
}
