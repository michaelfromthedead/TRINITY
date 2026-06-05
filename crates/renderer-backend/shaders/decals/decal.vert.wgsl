// SPDX-License-Identifier: MIT
//
// decal.vert.wgsl - Deferred Decal Vertex Shader (T-GPU-6.4)
//
// Renders decal projection boxes (unit cubes transformed by decal_to_world).
// Each decal is rendered as a box; the fragment shader reconstructs world
// position from depth and projects it into decal space to apply the decal.

// ---------------------------------------------------------------------------
// Uniform Structures
// ---------------------------------------------------------------------------

/// Global rendering parameters for the decal pass.
struct DecalParams {
    /// Combined view-projection matrix for camera.
    view_proj: mat4x4<f32>,
    /// Inverse view-projection for depth reconstruction.
    inv_view_proj: mat4x4<f32>,
    /// Camera world position (for view direction).
    camera_position: vec3<f32>,
    /// Padding to 16-byte alignment.
    _pad: f32,
}

/// Per-decal instance data.
/// Stored in GPU buffer, indexed by instance_index.
struct DecalInstance {
    /// Transform from world space to decal local space (unit cube).
    /// Used to determine if a fragment is inside the decal volume.
    world_to_decal: mat4x4<f32>,
    /// Transform from decal local space to world space.
    /// Used to transform the unit cube vertices and for normal transform.
    decal_to_world: mat4x4<f32>,
    /// Decal tint color (RGBA, alpha used for overall opacity).
    color: vec4<f32>,
    /// Atlas rectangle: xy = offset, zw = size (normalized 0-1).
    atlas_rect: vec4<f32>,
    /// Blend mode: 0=ALBEDO, 1=NORMAL, 2=BOTH, 3=EMISSIVE.
    blend_mode: u32,
    /// Normal map blend strength (0.0 = no normal, 1.0 = full normal).
    normal_strength: f32,
    /// Fade factor for lifetime-based fading (0.0 = invisible, 1.0 = full).
    fade: f32,
    /// Padding.
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Bind Groups
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> params: DecalParams;
@group(0) @binding(1) var<storage, read> decals: array<DecalInstance>;

// ---------------------------------------------------------------------------
// Vertex Output
// ---------------------------------------------------------------------------

struct VertexOutput {
    /// Clip-space position for rasterization.
    @builtin(position) clip_position: vec4<f32>,
    /// World-space position of the cube vertex.
    @location(0) world_position: vec3<f32>,
    /// Instance index to fetch decal data in fragment shader.
    @location(1) @interpolate(flat) instance_index: u32,
}

// ---------------------------------------------------------------------------
// Unit Cube Vertices
// ---------------------------------------------------------------------------

/// Generates unit cube vertex positions.
/// The cube spans [-0.5, 0.5] in all axes.
/// 36 vertices form 12 triangles (6 faces, 2 triangles each).
fn get_cube_vertex(index: u32) -> vec3<f32> {
    // Cube faces indexed for triangle strip-like pattern
    // Face order: -Z, +Z, -X, +X, -Y, +Y
    let positions = array<vec3<f32>, 36>(
        // -Z face (front)
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>(-0.5,  0.5, -0.5),
        // +Z face (back)
        vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5),
        vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5),
        // -X face (left)
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>(-0.5,  0.5,  0.5),
        // +X face (right)
        vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>( 0.5,  0.5, -0.5),
        // -Y face (bottom)
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5, -0.5),
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>(-0.5, -0.5, -0.5),
        // +Y face (top)
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5),
    );
    return positions[index % 36u];
}

// ---------------------------------------------------------------------------
// Vertex Shader Entry Point
// ---------------------------------------------------------------------------

@vertex
fn vs_decal(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    var output: VertexOutput;

    // Fetch decal instance data
    let decal = decals[iid];

    // Get unit cube vertex and transform to world space
    let local_pos = get_cube_vertex(vid);
    let world_pos = (decal.decal_to_world * vec4<f32>(local_pos, 1.0)).xyz;

    // Transform to clip space
    output.clip_position = params.view_proj * vec4<f32>(world_pos, 1.0);
    output.world_position = world_pos;
    output.instance_index = iid;

    return output;
}
