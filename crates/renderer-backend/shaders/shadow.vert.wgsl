// SPDX-License-Identifier: MIT
//
// shadow.vert.wgsl — Shadow map depth-only vertex shader (T-BRG-6.2).
//
// Transforms mesh vertices into the directional light's view-projection
// space for CSM depth rendering.

struct CascadeUniforms {
    light_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> cascade: CascadeUniforms;

struct ModelUniforms {
    model: mat4x4<f32>,
}

@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.clip_position = cascade.light_view_proj * world_pos;
    return output;
}
