// SPDX-License-Identifier: MIT
//
// pbr.vert.wgsl — PBR vertex shader (T-BRG-6.2).
//
// Transforms mesh vertices to clip space and passes world-space
// attributes (position, normal, tangent, texcoord) to the fragment
// shader for Cook-Torrance BRDF lighting.

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

struct ModelUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>, // mat3-packed-as-mat4 for alignment
    material_index: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec4<f32>,
    @location(3) texcoord: vec2<f32>,
    @location(4) material_index: u32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_projection * world_pos;
    output.world_position = world_pos.xyz;

    // Transform normal and tangent to world space using normal matrix.
    let n = normalize((model.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    let t = normalize((model.normal_matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
    output.world_normal = n;
    output.world_tangent = vec4<f32>(t, input.tangent.w);

    output.texcoord = input.texcoord;
    output.material_index = model.material_index;

    return output;
}
