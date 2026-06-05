// SPDX-License-Identifier: MIT
//
// trail.vert.wgsl - Trail Ribbon Vertex Shader (T-GPU-6.3)
//
// Renders ribbon geometry from trail points with Catmull-Rom interpolation.
// Each segment consists of 4 vertices forming a camera-facing ribbon strip.
//
// Features:
// - Camera-facing ribbon expansion (view-aligned)
// - Catmull-Rom tangent calculation for smooth curves
// - Width scaling along trail length
// - Alpha fade from head (newest) to tail (oldest)
// - UV modes: STRETCH (0-1 along length) or TILE (repeat based on distance)
//
// Algorithm:
// 1. Read trail point data from storage buffer
// 2. Calculate ribbon tangent using Catmull-Rom central difference
// 3. Expand perpendicular to tangent AND camera direction
// 4. Apply width scaling and age-based fade
// 5. Output UV coordinates based on texture mode

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;

// UV texture modes
const UV_MODE_STRETCH: u32 = 0u;  // Stretch texture along entire trail
const UV_MODE_TILE: u32 = 1u;     // Tile texture based on trail length

// Cap style constants (for future extension)
const CAP_STYLE_NONE: u32 = 0u;
const CAP_STYLE_ROUND: u32 = 1u;
const CAP_STYLE_FLAT: u32 = 2u;
const CAP_STYLE_ARROW: u32 = 3u;

// ============================================================================
// Data Structures
// ============================================================================

/// Uniform parameters for trail rendering.
struct TrailParams {
    /// Combined view-projection matrix.
    view_proj: mat4x4<f32>,
    /// Camera position in world space (for ribbon facing).
    camera_position: vec3<f32>,
    /// Base ribbon width in world units.
    ribbon_width: f32,
    /// Current time (for animated effects).
    time: f32,
    /// Age at which fade begins (0-1 normalized).
    fade_start: f32,
    /// Age at which fade ends (fully transparent).
    fade_end: f32,
    /// UV texture mode: 0=STRETCH, 1=TILE.
    uv_mode: u32,
    /// Tile repeat factor (for UV_MODE_TILE).
    tile_factor: f32,
    /// Total trail length (sum of segment distances).
    total_length: f32,
    /// Number of active trail points.
    point_count: u32,
    /// Cap style: 0=NONE, 1=ROUND, 2=FLAT, 3=ARROW.
    cap_style: u32,
}

/// Single point in the trail (matches Rust TrailPoint - 64 bytes).
struct TrailPoint {
    /// World-space position.
    position: vec3<f32>,
    /// Age of this point (seconds since creation).
    age: f32,
    /// Pre-computed tangent direction (normalized).
    direction: vec3<f32>,
    /// Width scale factor (0-1, multiplied with ribbon_width).
    width_scale: f32,
    /// Color at this point (RGBA).
    color: vec4<f32>,
    /// Distance from trail head to this point.
    distance_from_head: f32,
    /// Padding for alignment.
    _padding: vec3<f32>,
}

/// Vertex shader output.
struct VertexOutput {
    /// Clip-space position (required by GPU).
    @builtin(position) clip_position: vec4<f32>,
    /// Texture coordinates (u=along trail, v=across ribbon).
    @location(0) uv: vec2<f32>,
    /// Interpolated color.
    @location(1) color: vec4<f32>,
    /// Alpha factor (from age-based fade).
    @location(2) alpha: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: TrailParams;
@group(0) @binding(1) var<storage, read> trail_points: array<TrailPoint>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate Catmull-Rom tangent for a point given neighbors.
/// Returns normalized tangent direction.
fn catmull_rom_tangent(
    prev_pos: vec3<f32>,
    curr_pos: vec3<f32>,
    next_pos: vec3<f32>,
    has_prev: bool,
    has_next: bool
) -> vec3<f32> {
    var tangent: vec3<f32>;

    if has_prev && has_next {
        // Central difference (Catmull-Rom)
        tangent = (next_pos - prev_pos) * 0.5;
    } else if has_next {
        // Forward difference (first point)
        tangent = next_pos - curr_pos;
    } else if has_prev {
        // Backward difference (last point)
        tangent = curr_pos - prev_pos;
    } else {
        // Single point, use arbitrary direction
        tangent = vec3<f32>(0.0, 0.0, 1.0);
    }

    let len = length(tangent);
    if len > 0.0001 {
        return tangent / len;
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

/// Calculate ribbon expansion direction (perpendicular to tangent and camera view).
fn calculate_ribbon_right(
    position: vec3<f32>,
    tangent: vec3<f32>,
    camera_pos: vec3<f32>
) -> vec3<f32> {
    // Direction from position to camera
    let to_camera = camera_pos - position;
    let to_camera_len = length(to_camera);

    var view_dir: vec3<f32>;
    if to_camera_len > 0.0001 {
        view_dir = to_camera / to_camera_len;
    } else {
        view_dir = vec3<f32>(0.0, 1.0, 0.0);
    }

    // Right vector is perpendicular to both tangent and view direction
    var right = cross(tangent, view_dir);
    let right_len = length(right);

    if right_len < 0.0001 {
        // Degenerate case: tangent parallel to view
        // Use an arbitrary perpendicular
        let up = vec3<f32>(0.0, 1.0, 0.0);
        right = cross(tangent, up);
        let right_len2 = length(right);
        if right_len2 < 0.0001 {
            right = vec3<f32>(1.0, 0.0, 0.0);
        } else {
            right = right / right_len2;
        }
    } else {
        right = right / right_len;
    }

    return right;
}

/// Calculate alpha fade based on point age and fade parameters.
fn calculate_fade_alpha(age: f32, fade_start: f32, fade_end: f32) -> f32 {
    if fade_end <= fade_start {
        return 1.0;
    }

    if age <= fade_start {
        return 1.0;
    }

    if age >= fade_end {
        return 0.0;
    }

    // Linear interpolation in fade region
    return 1.0 - (age - fade_start) / (fade_end - fade_start);
}

/// Calculate UV.u coordinate based on mode.
fn calculate_u_coordinate(
    distance_from_head: f32,
    total_length: f32,
    uv_mode: u32,
    tile_factor: f32
) -> f32 {
    if uv_mode == UV_MODE_TILE {
        // Tile: repeat texture based on distance
        return distance_from_head * tile_factor;
    } else {
        // Stretch: 0 at head, 1 at tail
        if total_length > 0.0001 {
            return distance_from_head / total_length;
        }
        return 0.0;
    }
}

// ============================================================================
// Vertex Shader Entry Point
// ============================================================================

/// Trail ribbon vertex shader.
///
/// Vertex layout: Each trail segment uses 4 vertices (forming 2 triangles).
/// For N points, we have N-1 segments = (N-1)*4 vertices.
///
/// Vertex indices within segment:
///   0 = start left,  1 = start right
///   2 = end left,    3 = end right
///
/// Triangle strip order: 0, 1, 2, 3 forms two triangles.
@vertex
fn vs_trail(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var output: VertexOutput;

    let point_count = params.point_count;

    // Handle degenerate cases
    if point_count < 2u {
        // Not enough points for a trail
        output.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        output.uv = vec2<f32>(0.0, 0.0);
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.alpha = 0.0;
        return output;
    }

    // Calculate segment and vertex within segment
    // Each segment has 4 vertices (triangle strip)
    let segment_count = point_count - 1u;
    let segment_index = vid / 4u;
    let vertex_in_segment = vid % 4u;

    // Clamp segment index
    if segment_index >= segment_count {
        output.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        output.uv = vec2<f32>(0.0, 0.0);
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.alpha = 0.0;
        return output;
    }

    // Determine which point (start or end of segment)
    // 0,1 = segment start point; 2,3 = segment end point
    let use_end_point = vertex_in_segment >= 2u;

    // Determine left/right side
    // 0,2 = left side; 1,3 = right side
    let use_right_side = (vertex_in_segment & 1u) != 0u;

    // Get point indices
    let point_index = select(segment_index, segment_index + 1u, use_end_point);
    let point = trail_points[point_index];

    // Get neighbor points for Catmull-Rom tangent calculation
    let has_prev = point_index > 0u;
    let has_next = point_index < point_count - 1u;

    var prev_pos = point.position;
    var next_pos = point.position;

    if has_prev {
        prev_pos = trail_points[point_index - 1u].position;
    }
    if has_next {
        next_pos = trail_points[point_index + 1u].position;
    }

    // Calculate tangent (use pre-computed if available, otherwise compute)
    var tangent: vec3<f32>;
    let tangent_len = length(point.direction);
    if tangent_len > 0.5 {
        // Use pre-computed direction
        tangent = point.direction;
    } else {
        // Compute Catmull-Rom tangent
        tangent = catmull_rom_tangent(prev_pos, point.position, next_pos, has_prev, has_next);
    }

    // Calculate ribbon expansion direction
    let right = calculate_ribbon_right(point.position, tangent, params.camera_position);

    // Calculate vertex position
    let half_width = params.ribbon_width * point.width_scale * 0.5;
    let side_offset = select(-half_width, half_width, use_right_side);
    let world_position = point.position + right * side_offset;

    // Transform to clip space
    output.clip_position = params.view_proj * vec4<f32>(world_position, 1.0);

    // Calculate UV coordinates
    let u = calculate_u_coordinate(
        point.distance_from_head,
        params.total_length,
        params.uv_mode,
        params.tile_factor
    );
    let v = select(0.0, 1.0, use_right_side);
    output.uv = vec2<f32>(u, v);

    // Pass through color
    output.color = point.color;

    // Calculate alpha from age-based fade
    output.alpha = calculate_fade_alpha(point.age, params.fade_start, params.fade_end);

    return output;
}
