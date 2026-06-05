// Probe Blend Compute Shader
// Generated for TRINITY Engine - Reflection Probe System
// Configuration: max_probes=4, falloff=SMOOTH

// Constants
const MAX_PROBES: u32 = 4u;
const MIN_WEIGHT: f32 = 0.001000;
const BLEND_DISTANCE: f32 = 2.000000;
const NORMAL_WEIGHT: f32 = 0.300000;
const VISIBILITY_WEIGHT: f32 = 0.200000;
const EPSILON: f32 = 1e-6;

// Probe data structure
struct ProbeData {
    position: vec3<f32>,
    radius: f32,
    bounds_min: vec3<f32>,
    _pad0: f32,
    bounds_max: vec3<f32>,
    _pad1: f32,
};

// G-buffer input
struct GBufferSample {
    position: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
};

// Uniforms
struct Uniforms {
    camera_position: vec3<f32>,
    probe_count: u32,
    output_size: vec2<u32>,
    roughness_levels: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> probes: array<ProbeData>;
@group(0) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_material: texture_2d<f32>;
@group(0) @binding(5) var probe_cubemaps: texture_cube_array<f32>;
@group(0) @binding(6) var cubemap_sampler: sampler;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

// Check if point is inside probe bounds
fn point_in_probe(point: vec3<f32>, probe: ProbeData) -> bool {
    return all(point >= probe.bounds_min) && all(point <= probe.bounds_max);
}

// Calculate distance falloff
fn distance_falloff(distance: f32, max_dist: f32) -> f32 {
    let normalized_dist = min(distance / max_dist, 1.0);
    let blend_distance = BLEND_DISTANCE;
    return smoothstep(0.0, 1.0, 1.0 - normalized_dist);
}

// Calculate normal alignment factor
fn normal_alignment(normal: vec3<f32>, to_probe: vec3<f32>) -> f32 {
    let to_probe_norm = normalize(to_probe);
    let dot_product = dot(normal, to_probe_norm);
    return max(0.0, (dot_product + 1.0) * 0.5);
}

// Calculate probe influence weight
fn calculate_weight(
    position: vec3<f32>,
    normal: vec3<f32>,
    probe: ProbeData,
) -> f32 {
    if (!point_in_probe(position, probe)) {
        return 0.0;
    }

    let extent = probe.bounds_max - probe.bounds_min;
    let max_dist = length(extent) * 0.5;
    let to_probe = probe.position - position;
    let distance = length(to_probe);

    // Distance weight
    let dist_weight = distance_falloff(distance, max_dist);

    // Normal alignment
    let normal_align = normal_alignment(normal, to_probe);
    let normal_modifier = (1.0 - NORMAL_WEIGHT) + NORMAL_WEIGHT * normal_align;

    // Visibility assumed 1.0 (would need raytracing for occlusion)
    let vis_modifier = 1.0;

    return max(0.0, min(1.0, dist_weight * normal_modifier * vis_modifier));
}

// Sample cubemap with roughness
fn sample_probe(
    probe_index: u32,
    direction: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let mip_level = roughness * f32(uniforms.roughness_levels - 1u);
    return textureSampleLevel(
        probe_cubemaps,
        cubemap_sampler,
        direction,
        probe_index,
        mip_level
    ).rgb;
}

// Main compute kernel
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if (pixel.x >= uniforms.output_size.x || pixel.y >= uniforms.output_size.y) {
        return;
    }

    let output_index = pixel.y * uniforms.output_size.x + pixel.x;

    // Sample G-buffer
    let position = textureLoad(gbuffer_position, pixel, 0).xyz;
    let normal = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let material = textureLoad(gbuffer_material, pixel, 0);
    let roughness = material.x;
    let metallic = material.y;

    // Calculate reflection direction
    let view_dir = normalize(uniforms.camera_position - position);
    let reflect_dir = reflect(-view_dir, normal);

    // Collect probe influences
    var weights: array<f32, MAX_PROBES>;
    var probe_indices: array<u32, MAX_PROBES>;
    var weight_count: u32 = 0u;
    var total_weight: f32 = 0.0;

    for (var i: u32 = 0u; i < uniforms.probe_count && weight_count < MAX_PROBES; i = i + 1u) {
        let probe = probes[i];
        let weight = calculate_weight(position, normal, probe);

        if (weight >= MIN_WEIGHT) {
            weights[weight_count] = weight;
            probe_indices[weight_count] = i;
            total_weight = total_weight + weight;
            weight_count = weight_count + 1u;
        }
    }

    // Blend samples
    var result = vec3<f32>(0.0);

    if (weight_count > 0u) {
        let inv_total = select(1.0 / total_weight, 1.0, total_weight < EPSILON);

        for (var i: u32 = 0u; i < weight_count; i = i + 1u) {
            let normalized_weight = weights[i] * inv_total;
            let sample = sample_probe(probe_indices[i], reflect_dir, roughness);
            result = result + sample * normalized_weight;
        }
    }

    output[output_index] = vec4<f32>(result, 1.0);
}

