// SPDX-License-Identifier: MIT
//
// meshlet_render.vert.wgsl - Meshlet Rendering Vertex Shader (T-GPU-4.5)
//
// Renders visible meshlets using indirect draws. Reads from meshlet vertex/index
// buffers and transforms vertices via instance buffer lookup.
//
// Pipeline position:
// - Input: Meshlet vertex data, indirect draw commands from meshlet culling
// - Output: World position, normal, UV, tangent for fragment shading
//
// The vertex shader uses gl_InstanceIndex from indirect draws to look up
// the meshlet instance data and apply per-instance transforms.
//
// Performance Target: <0.1ms for 100K visible meshlets

// ============================================================================
// Structures
// ============================================================================

/// Camera/view uniforms.
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

/// Per-meshlet instance data from culling output.
/// Contains transform reference and material data.
struct MeshletInstance {
    /// Model matrix for this instance.
    model: mat4x4<f32>,
    /// Normal matrix (inverse transpose of model, packed as mat3 in vec4s).
    normal_matrix: mat4x4<f32>,
    /// Meshlet index within the mesh.
    meshlet_index: u32,
    /// Mesh ID for vertex/index buffer lookup.
    mesh_id: u32,
    /// Material ID for shading.
    material_id: u32,
    /// Instance flags (e.g., double-sided, cast shadows).
    flags: u32,
}

/// Meshlet render parameters.
struct MeshletRenderParams {
    /// Number of meshlets to render.
    num_meshlets: u32,
    /// Flags: bit 0 = visibility buffer mode, bit 1 = alpha test.
    flags: u32,
    /// Alpha test threshold (when alpha test is enabled).
    alpha_cutoff: f32,
    /// Reserved for future use.
    _pad: u32,
    /// Viewport dimensions for visibility ID packing.
    viewport_size: vec2<f32>,
    /// Near/far plane for depth encoding.
    near_plane: f32,
    far_plane: f32,
}

/// Meshlet vertex data (32 bytes).
struct MeshletVertex {
    /// Position in local space.
    position: vec3<f32>,
    /// Packed normal (octahedral encoding in u16x2 stored as f32).
    normal_packed: f32,
    /// Texture coordinates.
    texcoord: vec2<f32>,
    /// Packed tangent (xyz in octahedral encoding, w is sign).
    tangent_packed: vec2<f32>,
}

/// Meshlet descriptor referencing vertex/index buffers.
struct Meshlet {
    /// Offset into vertex index buffer (number of u32s).
    vertex_offset: u32,
    /// Offset into local triangle index buffer (bytes).
    triangle_offset: u32,
    /// Number of vertices in this meshlet (max 64).
    vertex_count: u32,
    /// Number of triangles in this meshlet (max 124).
    triangle_count: u32,
}

// ============================================================================
// Constants
// ============================================================================

/// Flag: Output to visibility buffer (pack instance+primitive ID).
const FLAG_VISIBILITY_BUFFER: u32 = 1u;
/// Flag: Perform alpha testing.
const FLAG_ALPHA_TEST: u32 = 2u;
/// Flag: Double-sided geometry (disable backface culling).
const FLAG_DOUBLE_SIDED: u32 = 4u;

// ============================================================================
// Bind Groups
// ============================================================================

// Group 0: Camera and render parameters
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: MeshletRenderParams;

// Group 1: Meshlet instance data (from culling output)
@group(1) @binding(0) var<storage, read> meshlet_instances: array<MeshletInstance>;

// Group 2: Meshlet geometry data
@group(2) @binding(0) var<storage, read> meshlet_vertices: array<MeshletVertex>;
@group(2) @binding(1) var<storage, read> meshlet_descriptors: array<Meshlet>;
@group(2) @binding(2) var<storage, read> vertex_indices: array<u32>;
@group(2) @binding(3) var<storage, read> local_indices: array<u32>;

// ============================================================================
// Vertex Input (from vertex buffer - positions only for indirect draws)
// ============================================================================

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

// ============================================================================
// Vertex Output
// ============================================================================

struct VertexOutput {
    /// Clip-space position.
    @builtin(position) clip_position: vec4<f32>,
    /// World-space position for lighting.
    @location(0) world_position: vec3<f32>,
    /// World-space normal (normalized).
    @location(1) world_normal: vec3<f32>,
    /// World-space tangent (xyz) and handedness (w).
    @location(2) world_tangent: vec4<f32>,
    /// Texture coordinates.
    @location(3) texcoord: vec2<f32>,
    /// Flat: Material ID for shading.
    @location(4) @interpolate(flat) material_id: u32,
    /// Flat: Instance ID for visibility buffer.
    @location(5) @interpolate(flat) instance_id: u32,
    /// Flat: Primitive ID (triangle index within meshlet).
    @location(6) @interpolate(flat) primitive_id: u32,
}

/// Visibility buffer output (reduced for depth+visibility pass).
struct VisibilityOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) visibility_id: u32,
}

// ============================================================================
// Helper Functions: Normal Encoding/Decoding
// ============================================================================

/// Decode octahedral-encoded normal from 32-bit packed value.
/// Uses SNorm16x2 encoding for high precision.
fn decode_octahedral_normal(packed: f32) -> vec3<f32> {
    // Unpack two SNorm16 values from f32 (reinterpreted as u32)
    let bits = bitcast<u32>(packed);
    let x_snorm = i32(bits & 0xFFFFu) - 32768;
    let y_snorm = i32(bits >> 16u) - 32768;

    // Convert to [-1, 1] range
    var oct = vec2<f32>(
        f32(x_snorm) / 32767.0,
        f32(y_snorm) / 32767.0
    );

    // Decode octahedral mapping
    var n = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));

    if (n.z < 0.0) {
        let sign_x = select(-1.0, 1.0, n.x >= 0.0);
        let sign_y = select(-1.0, 1.0, n.y >= 0.0);
        n = vec3<f32>(
            (1.0 - abs(oct.y)) * sign_x,
            (1.0 - abs(oct.x)) * sign_y,
            n.z
        );
    }

    return normalize(n);
}

/// Decode tangent from packed format.
/// xy: octahedral-encoded tangent direction, stored as two f32s.
fn decode_tangent(packed: vec2<f32>) -> vec4<f32> {
    // First component contains octahedral XY, second contains sign
    let tangent_dir = decode_octahedral_normal(packed.x);
    let sign_bits = bitcast<u32>(packed.y);
    let tangent_sign = select(-1.0, 1.0, (sign_bits & 1u) == 0u);

    return vec4<f32>(tangent_dir, tangent_sign);
}

/// Pack instance ID and primitive ID for visibility buffer.
/// Instance: 20 bits (max 1M instances)
/// Primitive: 12 bits (max 4K triangles)
fn pack_visibility_id(instance_id: u32, primitive_id: u32) -> u32 {
    return ((instance_id & 0xFFFFFu) << 12u) | (primitive_id & 0xFFFu);
}

// ============================================================================
// Main Vertex Shader: Full Output (G-Buffer Pass)
// ============================================================================

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Get instance data from indirect draw instance index
    let instance = meshlet_instances[input.instance_index];
    let meshlet = meshlet_descriptors[instance.meshlet_index];

    // Calculate triangle and vertex within triangle
    let triangle_id = input.vertex_index / 3u;
    let vertex_in_triangle = input.vertex_index % 3u;

    // Bounds check
    if (triangle_id >= meshlet.triangle_count) {
        // Return degenerate vertex for out-of-bounds
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }

    // Read local index (packed as bytes, 3 per triangle)
    let local_idx_byte_offset = meshlet.triangle_offset + triangle_id * 3u + vertex_in_triangle;
    let local_idx_word = local_indices[local_idx_byte_offset / 4u];
    let local_idx_shift = (local_idx_byte_offset % 4u) * 8u;
    let local_idx = (local_idx_word >> local_idx_shift) & 0xFFu;

    // Map local index to global vertex index
    let global_vertex_idx = vertex_indices[meshlet.vertex_offset + local_idx];

    // Read vertex data
    let vertex = meshlet_vertices[global_vertex_idx];

    // Decode normal and tangent
    let local_normal = decode_octahedral_normal(vertex.normal_packed);
    let local_tangent = decode_tangent(vertex.tangent_packed);

    // Transform to world space
    let world_pos = instance.model * vec4<f32>(vertex.position, 1.0);
    output.world_position = world_pos.xyz;

    // Transform normal using normal matrix (upper-left 3x3)
    let normal_mat = mat3x3<f32>(
        instance.normal_matrix[0].xyz,
        instance.normal_matrix[1].xyz,
        instance.normal_matrix[2].xyz
    );
    output.world_normal = normalize(normal_mat * local_normal);

    // Transform tangent
    let world_tangent_dir = normalize(normal_mat * local_tangent.xyz);
    output.world_tangent = vec4<f32>(world_tangent_dir, local_tangent.w);

    // Pass through texture coordinates
    output.texcoord = vertex.texcoord;

    // Transform to clip space
    output.clip_position = camera.view_projection * world_pos;

    // Pass instance metadata
    output.material_id = instance.material_id;
    output.instance_id = input.instance_index;
    output.primitive_id = triangle_id;

    return output;
}

// ============================================================================
// Visibility Pass Vertex Shader (Depth + Visibility ID only)
// ============================================================================

@vertex
fn vs_visibility(input: VertexInput) -> VisibilityOutput {
    var output: VisibilityOutput;

    // Get instance data
    let instance = meshlet_instances[input.instance_index];
    let meshlet = meshlet_descriptors[instance.meshlet_index];

    // Calculate triangle and vertex within triangle
    let triangle_id = input.vertex_index / 3u;
    let vertex_in_triangle = input.vertex_index % 3u;

    // Bounds check
    if (triangle_id >= meshlet.triangle_count) {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.visibility_id = 0xFFFFFFFFu;
        return output;
    }

    // Read local index
    let local_idx_byte_offset = meshlet.triangle_offset + triangle_id * 3u + vertex_in_triangle;
    let local_idx_word = local_indices[local_idx_byte_offset / 4u];
    let local_idx_shift = (local_idx_byte_offset % 4u) * 8u;
    let local_idx = (local_idx_word >> local_idx_shift) & 0xFFu;

    // Map to global vertex
    let global_vertex_idx = vertex_indices[meshlet.vertex_offset + local_idx];
    let vertex = meshlet_vertices[global_vertex_idx];

    // Transform to clip space
    let world_pos = instance.model * vec4<f32>(vertex.position, 1.0);
    output.clip_position = camera.view_projection * world_pos;

    // Pack visibility ID
    output.visibility_id = pack_visibility_id(input.instance_index, triangle_id);

    return output;
}

// ============================================================================
// Shadow Pass Vertex Shader (Depth only, no outputs except position)
// ============================================================================

struct ShadowOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_shadow(input: VertexInput) -> ShadowOutput {
    var output: ShadowOutput;

    // Get instance data
    let instance = meshlet_instances[input.instance_index];
    let meshlet = meshlet_descriptors[instance.meshlet_index];

    // Calculate triangle and vertex within triangle
    let triangle_id = input.vertex_index / 3u;
    let vertex_in_triangle = input.vertex_index % 3u;

    // Bounds check
    if (triangle_id >= meshlet.triangle_count) {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }

    // Read local index
    let local_idx_byte_offset = meshlet.triangle_offset + triangle_id * 3u + vertex_in_triangle;
    let local_idx_word = local_indices[local_idx_byte_offset / 4u];
    let local_idx_shift = (local_idx_byte_offset % 4u) * 8u;
    let local_idx = (local_idx_word >> local_idx_shift) & 0xFFu;

    // Map to global vertex
    let global_vertex_idx = vertex_indices[meshlet.vertex_offset + local_idx];
    let vertex = meshlet_vertices[global_vertex_idx];

    // Transform directly to clip space (light's view-projection expected in camera uniform)
    let world_pos = instance.model * vec4<f32>(vertex.position, 1.0);
    output.clip_position = camera.view_projection * world_pos;

    return output;
}
