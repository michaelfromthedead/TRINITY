// SPDX-License-Identifier: MIT
//
// virtual_lod.comp.wgsl - Virtual Geometry LOD System for TRINITY Engine (T-GPU-8.3)
//
// Nanite-style continuous LOD management with:
// - Screen-space error computation from geometric error
// - LOD bias calculation from camera distance
// - Dithered/alpha LOD transition blending
// - Streaming priority calculation (visibility + error + size)
// - Page residency tracking for virtual texturing integration
//
// One thread per mesh instance. Output: LOD selection, blend factor, streaming priority.
//
// Screen-space error formula:
//   screen_error = geometric_error * (fov_factor / distance) * screen_height
//
// LOD selection: find highest-quality LOD where screen_error < threshold
//
// Performance: O(n) work, single dispatch, <0.1ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Maximum LOD levels per virtual mesh.
const MAX_LOD_LEVELS: u32 = 16u;

/// Invalid LOD marker (mesh culled or not visible).
const INVALID_LOD: u32 = 0xFFFFFFFFu;

/// Flags for LODParams
const FLAG_USE_DITHER: u32 = 1u;        // Use dithered transition instead of alpha blend
const FLAG_FORCE_LOD: u32 = 2u;         // Use forced_lod from params (debug)
const FLAG_DISABLE_STREAMING: u32 = 4u;  // Don't compute streaming priority
const FLAG_PAGE_TRACKING: u32 = 8u;      // Enable page residency tracking

/// Streaming priority tiers
const PRIORITY_CRITICAL: u32 = 0u;  // Visible, large screen error (pop-in imminent)
const PRIORITY_HIGH: u32 = 1u;      // Visible, moderate error
const PRIORITY_NORMAL: u32 = 2u;    // Visible, low error
const PRIORITY_LOW: u32 = 3u;       // Barely visible or far away

// ============================================================================
// Structs
// ============================================================================

/// Virtual LOD system parameters (64 bytes, uniform buffer).
///
/// Memory Layout:
/// | Offset | Field              | Size |
/// |--------|-------------------|------|
/// | 0      | camera_position   | 12   |
/// | 12     | error_threshold   | 4    |
/// | 16     | lod_bias          | 4    |
/// | 20     | transition_width  | 4    |
/// | 24     | streaming_budget  | 4    |
/// | 28     | flags             | 4    |
/// | 32     | num_instances     | 4    |
/// | 36     | screen_height     | 4    |
/// | 40     | fov_y             | 4    |
/// | 44     | forced_lod        | 4    |
/// | 48     | frame_index       | 4    |
/// | 52     | _pad0             | 4    |
/// | 56     | _pad1             | 4    |
/// | 60     | _pad2             | 4    |
struct VirtualLODParams {
    /// Camera position in world space.
    camera_position: vec3<f32>,
    /// Screen-space error threshold in pixels.
    error_threshold: f32,
    /// Global LOD bias: negative = higher quality, positive = lower quality.
    lod_bias: f32,
    /// Dither/blend transition width (0-1 range for blend factor).
    transition_width: f32,
    /// Streaming budget in bytes per frame.
    streaming_budget: u32,
    /// Flags (see FLAG_* constants).
    flags: u32,
    /// Number of mesh instances to process.
    num_instances: u32,
    /// Screen height in pixels (for screen-space error).
    screen_height: f32,
    /// Vertical field of view in radians.
    fov_y: f32,
    /// Forced LOD level (when FLAG_FORCE_LOD set).
    forced_lod: u32,
    /// Frame index for temporal dithering.
    frame_index: u32,
    /// Padding for 16-byte alignment.
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// LOD level descriptor (16 bytes).
///
/// Memory Layout:
/// | Offset | Field           | Size |
/// |--------|-----------------|------|
/// | 0      | geometric_error | 4    |
/// | 4      | triangle_count  | 4    |
/// | 8      | vertex_offset   | 4    |
/// | 12     | index_offset    | 4    |
struct LODLevel {
    /// Maximum geometric error (deviation from full-res) in world units.
    geometric_error: f32,
    /// Number of triangles in this LOD level.
    triangle_count: u32,
    /// Byte offset to vertex data in the vertex buffer.
    vertex_offset: u32,
    /// Byte offset to index data in the index buffer.
    index_offset: u32,
}

/// Virtual mesh instance data (128 bytes).
///
/// Memory Layout:
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | position          | 12   |
/// | 12     | bounding_radius   | 4    |
/// | 16     | lod_levels        | 96   | (6 LODLevel entries)
/// | 112    | num_lods          | 4    |
/// | 116    | page_id           | 4    |
/// | 120    | mesh_id           | 4    |
/// | 124    | _pad              | 4    |
struct VirtualMesh {
    /// World position (bounding sphere center).
    position: vec3<f32>,
    /// Bounding sphere radius.
    bounding_radius: f32,
    /// LOD levels for this mesh (up to 6 levels inline).
    lod_levels: array<LODLevel, 6>,
    /// Number of valid LOD levels.
    num_lods: u32,
    /// Page ID for virtual texturing integration.
    page_id: u32,
    /// Unique mesh identifier.
    mesh_id: u32,
    /// Padding.
    _pad: u32,
}

/// Streaming priority request (16 bytes).
///
/// Memory Layout:
/// | Offset | Field     | Size |
/// |--------|-----------|------|
/// | 0      | priority  | 4    |
/// | 4      | mesh_id   | 4    |
/// | 8      | page_id   | 4    |
/// | 12     | lod_level | 4    |
struct StreamingPriority {
    /// Priority value (lower = more urgent). Packed: [tier:8][priority:24]
    priority: u32,
    /// Mesh ID for tracking.
    mesh_id: u32,
    /// Page ID for virtual texturing.
    page_id: u32,
    /// Target LOD level to stream.
    lod_level: u32,
}

/// LOD selection result (16 bytes).
///
/// Memory Layout:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | lod_level      | 4    |
/// | 4      | blend_factor   | 4    |
/// | 8      | screen_error   | 4    |
/// | 12     | flags          | 4    |
struct LODResult {
    /// Selected LOD level (0 = highest detail, INVALID_LOD = culled).
    lod_level: u32,
    /// Blend factor for transition (0.0 = fully primary, 1.0 = fully secondary).
    blend_factor: f32,
    /// Computed screen-space error in pixels.
    screen_error: f32,
    /// Result flags: bit 0 = needs_streaming, bit 1 = page_resident.
    flags: u32,
}

/// Page residency entry (8 bytes).
struct PageResidency {
    /// Page ID.
    page_id: u32,
    /// Residency status: bit 0 = resident, bits 1-7 = last access frame delta.
    status: u32,
}

// ============================================================================
// Bindings
// ============================================================================

/// LOD parameters (uniform buffer).
@group(0) @binding(0) var<uniform> params: VirtualLODParams;

/// Virtual mesh instances (read-only storage buffer).
@group(0) @binding(1) var<storage, read> meshes: array<VirtualMesh>;

/// LOD results (read-write storage buffer).
@group(0) @binding(2) var<storage, read_write> results: array<LODResult>;

/// Streaming priority queue (read-write storage buffer).
@group(0) @binding(3) var<storage, read_write> priorities: array<StreamingPriority>;

/// Page residency table (read-write storage buffer, optional).
@group(0) @binding(4) var<storage, read_write> page_residency: array<PageResidency>;

// ============================================================================
// Screen-Space Error Computation
// ============================================================================

/// Compute screen-space error from geometric error and distance.
///
/// The formula projects the geometric error sphere to screen space:
///   screen_error = geometric_error * (cot(fov/2) / distance) * (screen_height / 2)
///
/// This gives the error in pixels that would result from using a simplified LOD.
fn compute_screen_error(
    geometric_error: f32,
    distance: f32,
    fov_y: f32,
    screen_height: f32
) -> f32 {
    // Guard against division by zero
    if (distance <= 0.001) {
        return 1000000.0; // Very large error (force highest LOD)
    }

    // fov_factor = cot(fov/2) = 1/tan(fov/2)
    let half_fov = fov_y * 0.5;
    let fov_factor = 1.0 / tan(half_fov);

    // Project geometric error to screen space
    // geometric_error is the maximum deviation in world units
    // At distance d, this subtends an angle of atan(error/d) ~ error/d for small angles
    // Multiply by screen height and fov factor to get pixels
    let screen_error = geometric_error * fov_factor * screen_height * 0.5 / distance;

    return screen_error;
}

/// Compute screen size (coverage) for streaming priority.
fn compute_screen_size(
    bounding_radius: f32,
    distance: f32,
    fov_y: f32,
    screen_height: f32
) -> f32 {
    if (distance <= 0.001) {
        return screen_height; // Very close = full screen
    }

    let half_fov = fov_y * 0.5;
    let fov_factor = 1.0 / tan(half_fov);

    // Projected diameter in pixels
    let screen_size = bounding_radius * 2.0 * fov_factor * screen_height * 0.5 / distance;

    return clamp(screen_size, 0.0, screen_height);
}

// ============================================================================
// LOD Selection
// ============================================================================

/// Select the appropriate LOD level based on screen-space error threshold.
///
/// Uses binary search to find the highest-quality LOD (lowest index) whose
/// screen-space error is below the threshold. The LOD bias shifts the effective
/// threshold: positive bias = allow more error = lower quality.
///
/// Returns: (lod_level, screen_error_at_that_lod)
fn select_lod(mesh: VirtualMesh, distance: f32, threshold: f32, bias: f32) -> vec2<f32> {
    // Apply bias to threshold: bias of +1 doubles acceptable error
    let biased_threshold = threshold * pow(2.0, bias);

    var selected_lod = 0u;
    var selected_error = 0.0;

    // Linear search from highest quality (LOD 0) to lowest
    // Find first LOD where screen error is acceptable
    for (var lod = 0u; lod < mesh.num_lods; lod++) {
        let level = mesh.lod_levels[lod];
        let screen_error = compute_screen_error(
            level.geometric_error,
            distance,
            params.fov_y,
            params.screen_height
        );

        selected_lod = lod;
        selected_error = screen_error;

        // If error is below threshold, this LOD is acceptable
        if (screen_error <= biased_threshold) {
            break;
        }
    }

    return vec2<f32>(f32(selected_lod), selected_error);
}

// ============================================================================
// LOD Transition Blending
// ============================================================================

/// Compute blend factor for smooth LOD transitions.
///
/// The blend factor indicates how much to blend between the current LOD
/// and the next lower-quality LOD.
///
/// For dithered transitions, this factor is compared against a dither pattern.
/// For alpha blending, both LODs are rendered with alpha = (1-factor, factor).
fn compute_blend_factor(
    screen_error: f32,
    threshold: f32,
    transition_width: f32
) -> f32 {
    if (transition_width <= 0.0) {
        return 0.0; // No blending
    }

    // Blend region: [threshold - width, threshold]
    // When error is at (threshold - width), blend_factor = 0.0 (use current LOD)
    // When error is at threshold, blend_factor = 1.0 (transition to next LOD)
    let blend_start = threshold * (1.0 - transition_width);
    let blend_end = threshold;

    if (screen_error <= blend_start) {
        return 0.0;
    }
    if (screen_error >= blend_end) {
        return 1.0;
    }

    // Linear interpolation in blend region
    return (screen_error - blend_start) / (blend_end - blend_start);
}

/// Generate dither threshold for temporal/spatial dithering.
fn dither_threshold(pixel_pos: vec2<u32>, frame_index: u32) -> f32 {
    // 4x4 Bayer dither matrix
    let bayer: array<f32, 16> = array<f32, 16>(
        0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
        12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0,
        3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
        15.0/16.0, 7.0/16.0, 13.0/16.0,  5.0/16.0
    );

    // Add temporal jitter using frame index
    let x = (pixel_pos.x + frame_index) % 4u;
    let y = (pixel_pos.y + frame_index / 4u) % 4u;
    let idx = y * 4u + x;

    return bayer[idx];
}

// ============================================================================
// Streaming Priority Computation
// ============================================================================

/// Compute streaming priority for a mesh instance.
///
/// Priority is based on:
/// - Visibility: visible meshes have higher priority
/// - Screen error: higher error = more urgent (pop-in visible)
/// - Screen size: larger objects are more noticeable
/// - Page residency: non-resident pages are more urgent
///
/// Lower priority value = more urgent.
fn compute_streaming_priority(
    mesh: VirtualMesh,
    distance: f32,
    screen_error: f32,
    selected_lod: u32,
    is_visible: bool
) -> StreamingPriority {
    var priority: StreamingPriority;
    priority.mesh_id = mesh.mesh_id;
    priority.page_id = mesh.page_id;
    priority.lod_level = selected_lod;

    if (!is_visible) {
        // Non-visible: lowest priority
        priority.priority = (PRIORITY_LOW << 24u) | 0xFFFFFFu;
        return priority;
    }

    // Compute urgency based on screen error relative to threshold
    let error_ratio = screen_error / max(params.error_threshold, 0.001);

    // Compute screen size for importance weighting
    let screen_size = compute_screen_size(
        mesh.bounding_radius,
        distance,
        params.fov_y,
        params.screen_height
    );
    let size_factor = clamp(screen_size / params.screen_height, 0.0, 1.0);

    // Combine factors: urgency = error_ratio * size_factor
    // Higher urgency = lower priority value
    let urgency = error_ratio * size_factor;

    // Determine tier based on urgency
    var tier = PRIORITY_LOW;
    if (urgency > 0.8) {
        tier = PRIORITY_CRITICAL;
    } else if (urgency > 0.5) {
        tier = PRIORITY_HIGH;
    } else if (urgency > 0.2) {
        tier = PRIORITY_NORMAL;
    }

    // Pack priority: [tier:8][inverse_urgency:24]
    // Inverse urgency so lower value = higher priority within tier
    let inverse_urgency = u32((1.0 - clamp(urgency, 0.0, 0.999)) * 16777215.0);
    priority.priority = (tier << 24u) | inverse_urgency;

    return priority;
}

// ============================================================================
// Main Compute Kernels
// ============================================================================

/// Main LOD selection and streaming priority computation.
///
/// Each thread processes one mesh instance:
/// 1. Compute distance from camera
/// 2. Select appropriate LOD based on screen-space error
/// 3. Compute blend factor for smooth transitions
/// 4. Compute streaming priority for data loading
/// 5. Update page residency tracking (if enabled)
@compute @workgroup_size(256)
fn compute_virtual_lod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= params.num_instances) {
        return;
    }

    let mesh = meshes[idx];

    // Compute distance from camera to mesh center
    let to_mesh = mesh.position - params.camera_position;
    let distance = length(to_mesh);

    // Check for forced LOD (debug mode)
    if ((params.flags & FLAG_FORCE_LOD) != 0u) {
        let forced = min(params.forced_lod, mesh.num_lods - 1u);
        let screen_error = compute_screen_error(
            mesh.lod_levels[forced].geometric_error,
            distance,
            params.fov_y,
            params.screen_height
        );
        results[idx] = LODResult(forced, 0.0, screen_error, 0u);
        return;
    }

    // Select LOD based on screen-space error
    let lod_result = select_lod(mesh, distance, params.error_threshold, params.lod_bias);
    let selected_lod = u32(lod_result.x);
    let screen_error = lod_result.y;

    // Compute blend factor for transitions
    let blend_factor = compute_blend_factor(
        screen_error,
        params.error_threshold,
        params.transition_width
    );

    // Determine result flags
    var result_flags = 0u;
    let is_visible = selected_lod != INVALID_LOD;

    // Compute streaming priority if enabled
    if ((params.flags & FLAG_DISABLE_STREAMING) == 0u) {
        let priority = compute_streaming_priority(
            mesh,
            distance,
            screen_error,
            selected_lod,
            is_visible
        );
        priorities[idx] = priority;

        // Check if streaming is needed (error above threshold)
        if (screen_error > params.error_threshold * 0.8) {
            result_flags |= 1u; // needs_streaming flag
        }
    }

    // Page residency tracking if enabled
    if ((params.flags & FLAG_PAGE_TRACKING) != 0u && mesh.page_id != 0xFFFFFFFFu) {
        let page_idx = mesh.page_id;
        let residency = page_residency[page_idx];

        // Check if page is resident
        if ((residency.status & 1u) != 0u) {
            result_flags |= 2u; // page_resident flag
        }

        // Update last access frame (atomic would be better, but simplified here)
        let frame_delta = params.frame_index & 0x7Fu;
        page_residency[page_idx].status = (residency.status & 1u) | (frame_delta << 1u);
    }

    // Store result
    results[idx] = LODResult(selected_lod, blend_factor, screen_error, result_flags);
}

/// Screen-space error only computation (no LOD selection).
/// Useful for debugging or when LOD levels are handled elsewhere.
@compute @workgroup_size(256)
fn compute_screen_error_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let mesh = meshes[idx];
    let to_mesh = mesh.position - params.camera_position;
    let distance = length(to_mesh);

    // Compute error for LOD 0 (highest detail)
    let screen_error = compute_screen_error(
        mesh.lod_levels[0].geometric_error,
        distance,
        params.fov_y,
        params.screen_height
    );

    results[idx] = LODResult(0u, 0.0, screen_error, 0u);
}

/// Streaming priority only (assumes LOD already selected).
/// Reads LOD from results buffer and computes priority.
@compute @workgroup_size(256)
fn compute_streaming_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let mesh = meshes[idx];
    let existing_result = results[idx];

    let to_mesh = mesh.position - params.camera_position;
    let distance = length(to_mesh);

    let is_visible = existing_result.lod_level != INVALID_LOD;

    let priority = compute_streaming_priority(
        mesh,
        distance,
        existing_result.screen_error,
        existing_result.lod_level,
        is_visible
    );

    priorities[idx] = priority;
}

/// LOD transition dither check.
/// Returns 1 if the mesh should render the secondary (lower-quality) LOD based on dither.
/// Used for screen-space dithered LOD transitions.
@compute @workgroup_size(256)
fn compute_dither_lod(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    var result = results[idx];

    // Only apply dithering if blending is active and dither flag is set
    if ((params.flags & FLAG_USE_DITHER) != 0u && result.blend_factor > 0.0 && result.blend_factor < 1.0) {
        // Use instance index to generate pseudo-random screen position
        // In practice, this would use actual screen position from the mesh
        let pseudo_pos = vec2<u32>(idx % 256u, idx / 256u);
        let dither = dither_threshold(pseudo_pos, params.frame_index);

        // If dither threshold > blend factor, use secondary LOD
        if (dither > result.blend_factor) {
            let mesh = meshes[idx];
            if (result.lod_level + 1u < mesh.num_lods) {
                result.lod_level = result.lod_level + 1u;
                // Set blend_factor to 1.0 to indicate we switched
                result.blend_factor = 1.0;
            }
        } else {
            result.blend_factor = 0.0; // Using primary LOD
        }

        results[idx] = result;
    }
}
