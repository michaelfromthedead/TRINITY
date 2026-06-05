// SPDX-License-Identifier: MIT
//
// gpu_visibility_read.comp.wgsl - Visibility Buffer Read (T-GPU-3.6)
//
// Reads visibility buffer and outputs material inputs for shading.
// Reconstructs world position, normals, and UVs from barycentrics.
//
// This shader is the core of deferred material evaluation:
// 1. Read visibility data (instance ID, primitive ID, barycentrics)
// 2. Look up triangle vertices from index/vertex buffers
// 3. Interpolate vertex attributes using barycentric coordinates
// 4. Transform to world space using instance transforms
// 5. Output shading inputs for material evaluation
//
// Performance Target: <0.2ms for 1080p (2M pixels) at 16x16 tiles

// ============================================================================
// Constants
// ============================================================================

/// Workgroup tile size: 16x16 = 256 threads
const TILE_SIZE: u32 = 16u;

/// Invalid instance ID marker (background pixels)
const INVALID_INSTANCE: u32 = 0xFFFFFFFFu;

/// Invalid primitive ID marker
const INVALID_PRIMITIVE: u32 = 0xFFFFFFFFu;

/// Small epsilon for normalization
const EPSILON: f32 = 1e-7;

// ============================================================================
// Structures
// ============================================================================

/// Parameters for visibility buffer read pass.
struct VisibilityReadParams {
    /// Screen width in pixels.
    screen_width: u32,
    /// Screen height in pixels.
    screen_height: u32,
    /// Tile offset X for tiled dispatch.
    tile_offset_x: u32,
    /// Tile offset Y for tiled dispatch.
    tile_offset_y: u32,
}

/// Visibility data per pixel (output from visibility pass).
struct VisibilityData {
    /// Instance ID (INVALID_INSTANCE = background).
    instance_id: u32,
    /// Primitive/triangle ID within the mesh.
    primitive_id: u32,
    /// Barycentric coordinates (beta, gamma); alpha = 1 - beta - gamma.
    barycentrics: vec2<f32>,
}

/// Per-instance transform data.
struct InstanceTransform {
    /// World matrix (model-to-world transform).
    world_matrix: mat4x4<f32>,
    /// Normal matrix (transpose inverse of world matrix upper-left 3x3).
    /// Used for correct normal transformation.
    normal_matrix_col0: vec3<f32>,
    _pad0: f32,
    normal_matrix_col1: vec3<f32>,
    _pad1: f32,
    normal_matrix_col2: vec3<f32>,
    _pad2: f32,
}

/// Vertex data for triangle reconstruction.
struct VertexData {
    /// Object-space position.
    position: vec3<f32>,
    _pad0: f32,
    /// Object-space normal (unit vector).
    normal: vec3<f32>,
    _pad1: f32,
    /// Texture coordinates.
    uv: vec2<f32>,
    /// Padding for alignment.
    _pad2: vec2<f32>,
    /// Tangent with handedness in w component.
    /// w = +1 or -1 indicates handedness for bitangent calculation.
    tangent: vec4<f32>,
}

/// Instance metadata for vertex/index buffer lookups.
struct InstanceMetadata {
    /// Base index into the global index buffer.
    index_offset: u32,
    /// Base vertex offset into the global vertex buffer.
    vertex_offset: u32,
    /// Material ID for this instance.
    material_id: u32,
    /// Reserved for future use.
    _pad: u32,
}

/// Output shading inputs for material evaluation.
struct ShadingInput {
    /// World-space position.
    world_pos: vec3<f32>,
    _pad0: f32,
    /// World-space unit normal.
    world_normal: vec3<f32>,
    _pad1: f32,
    /// Interpolated texture coordinates.
    uv: vec2<f32>,
    /// Instance ID for additional lookups.
    instance_id: u32,
    /// Material ID for material table lookup.
    material_id: u32,
    /// World-space unit tangent.
    tangent: vec3<f32>,
    _pad2: f32,
    /// World-space unit bitangent.
    bitangent: vec3<f32>,
    _pad3: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: VisibilityReadParams;

/// Visibility buffer (one entry per pixel).
@group(0) @binding(1) var<storage, read> visibility_buffer: array<VisibilityData>;

/// Per-instance transforms.
@group(0) @binding(2) var<storage, read> instance_transforms: array<InstanceTransform>;

/// Per-instance metadata (index/vertex offsets, material ID).
@group(0) @binding(3) var<storage, read> instance_metadata: array<InstanceMetadata>;

/// Global vertex buffer.
@group(0) @binding(4) var<storage, read> vertex_buffer: array<VertexData>;

/// Global index buffer.
@group(0) @binding(5) var<storage, read> index_buffer: array<u32>;

/// Output shading inputs (one per pixel).
@group(0) @binding(6) var<storage, read_write> shading_inputs: array<ShadingInput>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Interpolate a vec3 using barycentric coordinates.
/// bary.x = beta (weight for v1), bary.y = gamma (weight for v2)
/// alpha = 1 - beta - gamma (weight for v0)
fn interpolate_vec3(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, bary: vec2<f32>) -> vec3<f32> {
    let alpha = 1.0 - bary.x - bary.y;
    return v0 * alpha + v1 * bary.x + v2 * bary.y;
}

/// Interpolate a vec2 using barycentric coordinates.
fn interpolate_vec2(v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, bary: vec2<f32>) -> vec2<f32> {
    let alpha = 1.0 - bary.x - bary.y;
    return v0 * alpha + v1 * bary.x + v2 * bary.y;
}

/// Interpolate a vec4 using barycentric coordinates.
fn interpolate_vec4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, bary: vec2<f32>) -> vec4<f32> {
    let alpha = 1.0 - bary.x - bary.y;
    return v0 * alpha + v1 * bary.x + v2 * bary.y;
}

/// Normalize a vector, handling denormalized inputs gracefully.
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq < EPSILON * EPSILON) {
        return vec3<f32>(0.0, 1.0, 0.0); // Default up vector
    }
    return v * inverseSqrt(len_sq);
}

/// Transform position from object space to world space.
fn transform_position(pos: vec3<f32>, world_matrix: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_matrix * vec4<f32>(pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Transform normal from object space to world space using normal matrix.
fn transform_normal(normal: vec3<f32>, transform: InstanceTransform) -> vec3<f32> {
    // Reconstruct normal matrix from columns
    let normal_matrix = mat3x3<f32>(
        transform.normal_matrix_col0,
        transform.normal_matrix_col1,
        transform.normal_matrix_col2
    );
    return safe_normalize(normal_matrix * normal);
}

/// Transform tangent vector using the world matrix upper-left 3x3.
fn transform_tangent(tangent: vec3<f32>, world_matrix: mat4x4<f32>) -> vec3<f32> {
    let world_tangent = mat3x3<f32>(
        world_matrix[0].xyz,
        world_matrix[1].xyz,
        world_matrix[2].xyz
    ) * tangent;
    return safe_normalize(world_tangent);
}

/// Tangent space result for returning both tangent and bitangent.
struct TangentSpaceResult {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

/// Compute orthonormal tangent space from normal and tangent.
/// Handles non-orthogonal input via Gram-Schmidt orthogonalization.
fn compute_tangent_space(
    world_normal: vec3<f32>,
    interpolated_tangent: vec4<f32>,
    world_matrix: mat4x4<f32>
) -> TangentSpaceResult {
    // Transform tangent to world space
    var world_tangent = transform_tangent(interpolated_tangent.xyz, world_matrix);

    // Gram-Schmidt orthogonalization: remove normal component from tangent
    world_tangent = safe_normalize(world_tangent - world_normal * dot(world_normal, world_tangent));

    // Compute bitangent using handedness from tangent.w
    let bitangent = cross(world_normal, world_tangent) * interpolated_tangent.w;

    return TangentSpaceResult(world_tangent, safe_normalize(bitangent));
}

/// Calculate pixel index from 2D coordinates.
fn pixel_index(x: u32, y: u32) -> u32 {
    return y * params.screen_width + x;
}

/// Create an empty/invalid shading input for background pixels.
fn empty_shading_input() -> ShadingInput {
    return ShadingInput(
        vec3<f32>(0.0, 0.0, 0.0), 0.0,  // world_pos
        vec3<f32>(0.0, 1.0, 0.0), 0.0,  // world_normal
        vec2<f32>(0.0, 0.0),            // uv
        INVALID_INSTANCE,               // instance_id
        0u,                             // material_id
        vec3<f32>(1.0, 0.0, 0.0), 0.0,  // tangent
        vec3<f32>(0.0, 0.0, 1.0), 0.0   // bitangent
    );
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute @workgroup_size(16, 16)
fn visibility_read(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Calculate pixel coordinates with tile offset
    let pixel_x = gid.x + params.tile_offset_x * TILE_SIZE;
    let pixel_y = gid.y + params.tile_offset_y * TILE_SIZE;

    // Bounds check
    if (pixel_x >= params.screen_width || pixel_y >= params.screen_height) {
        return;
    }

    let pixel_idx = pixel_index(pixel_x, pixel_y);

    // Read visibility data
    let vis = visibility_buffer[pixel_idx];

    // Handle background pixels
    if (vis.instance_id == INVALID_INSTANCE || vis.primitive_id == INVALID_PRIMITIVE) {
        shading_inputs[pixel_idx] = empty_shading_input();
        return;
    }

    // Get instance data
    let transform = instance_transforms[vis.instance_id];
    let metadata = instance_metadata[vis.instance_id];

    // Calculate triangle vertex indices
    let tri_base = metadata.index_offset + vis.primitive_id * 3u;
    let i0 = index_buffer[tri_base + 0u] + metadata.vertex_offset;
    let i1 = index_buffer[tri_base + 1u] + metadata.vertex_offset;
    let i2 = index_buffer[tri_base + 2u] + metadata.vertex_offset;

    // Fetch triangle vertices
    let v0 = vertex_buffer[i0];
    let v1 = vertex_buffer[i1];
    let v2 = vertex_buffer[i2];

    // Interpolate vertex attributes using barycentrics
    let bary = vis.barycentrics;
    let obj_pos = interpolate_vec3(v0.position, v1.position, v2.position, bary);
    let obj_normal = interpolate_vec3(v0.normal, v1.normal, v2.normal, bary);
    let interp_uv = interpolate_vec2(v0.uv, v1.uv, v2.uv, bary);
    let interp_tangent = interpolate_vec4(v0.tangent, v1.tangent, v2.tangent, bary);

    // Transform to world space
    let world_pos = transform_position(obj_pos, transform.world_matrix);
    let world_normal = transform_normal(obj_normal, transform);

    // Compute orthonormal tangent space
    let tangent_space = compute_tangent_space(world_normal, interp_tangent, transform.world_matrix);

    // Write shading input
    shading_inputs[pixel_idx] = ShadingInput(
        world_pos, 0.0,
        world_normal, 0.0,
        interp_uv,
        vis.instance_id,
        metadata.material_id,
        tangent_space.tangent, 0.0,
        tangent_space.bitangent, 0.0
    );
}

// ============================================================================
// Single-Tile Variant (for small screens or final tiles)
// ============================================================================

@compute @workgroup_size(16, 16)
fn visibility_read_single_tile(@builtin(global_invocation_id) gid: vec3<u32>) {
    // No tile offset - direct pixel mapping
    let pixel_x = gid.x;
    let pixel_y = gid.y;

    if (pixel_x >= params.screen_width || pixel_y >= params.screen_height) {
        return;
    }

    let pixel_idx = pixel_index(pixel_x, pixel_y);
    let vis = visibility_buffer[pixel_idx];

    if (vis.instance_id == INVALID_INSTANCE || vis.primitive_id == INVALID_PRIMITIVE) {
        shading_inputs[pixel_idx] = empty_shading_input();
        return;
    }

    let transform = instance_transforms[vis.instance_id];
    let metadata = instance_metadata[vis.instance_id];

    let tri_base = metadata.index_offset + vis.primitive_id * 3u;
    let i0 = index_buffer[tri_base + 0u] + metadata.vertex_offset;
    let i1 = index_buffer[tri_base + 1u] + metadata.vertex_offset;
    let i2 = index_buffer[tri_base + 2u] + metadata.vertex_offset;

    let v0 = vertex_buffer[i0];
    let v1 = vertex_buffer[i1];
    let v2 = vertex_buffer[i2];

    let bary = vis.barycentrics;
    let obj_pos = interpolate_vec3(v0.position, v1.position, v2.position, bary);
    let obj_normal = interpolate_vec3(v0.normal, v1.normal, v2.normal, bary);
    let interp_uv = interpolate_vec2(v0.uv, v1.uv, v2.uv, bary);
    let interp_tangent = interpolate_vec4(v0.tangent, v1.tangent, v2.tangent, bary);

    let world_pos = transform_position(obj_pos, transform.world_matrix);
    let world_normal = transform_normal(obj_normal, transform);
    let tangent_space = compute_tangent_space(world_normal, interp_tangent, transform.world_matrix);

    shading_inputs[pixel_idx] = ShadingInput(
        world_pos, 0.0,
        world_normal, 0.0,
        interp_uv,
        vis.instance_id,
        metadata.material_id,
        tangent_space.tangent, 0.0,
        tangent_space.bitangent, 0.0
    );
}

// ============================================================================
// Clear Shading Inputs (Initialize to Invalid)
// ============================================================================

@compute @workgroup_size(256)
fn clear_shading_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_count = params.screen_width * params.screen_height;

    if (gid.x >= pixel_count) {
        return;
    }

    shading_inputs[gid.x] = empty_shading_input();
}
