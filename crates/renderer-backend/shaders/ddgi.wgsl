// SPDX-License-Identifier: MIT
//
// ddgi.wgsl — Dynamic Diffuse Global Illumination compute shaders (T-BRG-9.3).
//
// Two compute entry points:
//   1. ddgi_update_probes — shoots rays to update SH coefficients at each probe
//   2. ddgi_sample_probes — samples probe grid to produce indirect irradiance

// ── Constants ──

const SH_L0: f32 = 0.28209479177387814;  // 1 / (2 * sqrt(PI))
const SH_L1: f32 = 0.4886025119029199;   // sqrt(3 / (4 * PI))

const MAX_RAYS_PER_PROBE: u32 = 256u;
const MAX_PROBES: u32 = 4096u;
const PI: f32 = 3.14159265359;
const MAX_RAY_DISTANCE: f32 = 50.0;
const NUM_FRAMES_PER_UPDATE: u32 = 8u;

// ── Data structures ──

struct DDGIProbe {
    // Spherical harmonic coefficients: L0 (1 float) + L1 (3 floats) per RGB channel.
    // Packed as: sh_r[4], sh_g[4], sh_b[4].
    sh_r: vec4<f32>,  // [L0, L1.x, L1.y, L1.z] for red
    sh_g: vec4<f32>,  // [L0, L1.x, L1.y, L1.z] for green
    sh_b: vec4<f32>,  // [L0, L1.x, L1.y, L1.z] for blue
}

struct ProbeVolume {
    origin: vec3<f32>,
    _pad0: f32,
    extents: vec3<u32>,   // probes per axis
    spacing: f32,
    num_rays_per_probe: u32,
    max_ray_distance: f32,
    energy_preservation: f32,
    num_irradiance_texels: u32,
    num_depth_texels: u32,
    _pad1: f32,
    _pad2: f32,
}

// ── Uniforms and buffers ──

@group(0) @binding(0) var<uniform> volume: ProbeVolume;
@group(0) @binding(1) var<storage, read_write> probes: array<DDGIProbe, MAX_PROBES>;
@group(0) @binding(2) var<storage, read> ray_directions: array<vec3<f32>>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;
@group(0) @binding(4) var<uniform> frame_index: u32;

// Irradiance output (from sample pass).
@group(0) @binding(5) var irradiance_texture: texture_storage_2d<rgba16float, write>;

// World-space position and normal textures (from G-buffer for sample pass).
@group(0) @binding(6) var world_position_texture: texture_2d<f32>;
@group(0) @binding(7) var world_normal_texture: texture_2d<f32>;

// Camera uniforms for ray origin.
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(8) var<uniform> camera: CameraUniforms;

// ── Spherical harmonic evaluation ──

/// Evaluates SH at direction `dir` using L0 + L1 coefficients for a single channel.
fn eval_sh(channel_coeffs: vec4<f32>, dir: vec3<f32>) -> f32 {
    // L0 term.
    let l0 = SH_L0 * channel_coeffs.x;
    // L1 terms: Y(1,-1)=L1*y, Y(1,0)=L1*z, Y(1,1)=L1*x
    let l1 = SH_L1 * (channel_coeffs.y * dir.x + channel_coeffs.z * dir.y + channel_coeffs.w * dir.z);
    return l0 + l1;
}

/// Evaluates full RGB SH at direction `dir`.
fn eval_sh_rgb(probe: DDGIProbe, dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        eval_sh(probe.sh_r, dir),
        eval_sh(probe.sh_g, dir),
        eval_sh(probe.sh_b, dir),
    );
}

/// Projects irradiance into SH coefficients (additive accumulation).
fn project_sh(dir: vec3<f32>, irradiance: vec3<f32>) -> vec4<f32> {
    // Returns (sh_update for single channel); called per channel.
    let l0 = SH_L0;
    let l1 = SH_L1 * dir;
    return vec4<f32>(l0 * irradiance.x, l1.x * irradiance.x, l1.y * irradiance.x, l1.z * irradiance.x);
}

// ── Spatial indexing ──

/// Converts 3D probe grid index to linear probe index.
fn probe_index_3d_to_1d(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * volume.extents.x * volume.extents.y + iy * volume.extents.x + ix;
}

/// Converts linear probe index to world-space position.
fn probe_world_position(probe_idx: u32) -> vec3<f32> {
    let iz = probe_idx / (volume.extents.x * volume.extents.y);
    let remainder = probe_idx % (volume.extents.x * volume.extents.y);
    let iy = remainder / volume.extents.x;
    let ix = remainder % volume.extents.x;

    return volume.origin + vec3<f32>(
        f32(ix) * volume.spacing,
        f32(iy) * volume.spacing,
        f32(iz) * volume.spacing,
    );
}

// ── Entry point 1: Update probes ──

/// Updates SH coefficients for a subset of probes each frame.
/// Dispatched with workgroup dimension to match probe update count.
@compute @workgroup_size(8, 8, 1)
fn ddgi_update_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_probes = volume.extents.x * volume.extents.y * volume.extents.z;
    let probes_per_frame = max(total_probes / NUM_FRAMES_PER_UPDATE, 1u);

    let probe_offset = (frame_index % NUM_FRAMES_PER_UPDATE) * probes_per_frame;
    let local_idx = gid.x + gid.y * 64u; // 64-wide rows

    if local_idx >= probes_per_frame {
        return;
    }

    let probe_idx = probe_offset + local_idx;
    if probe_idx >= total_probes {
        return;
    }

    let probe_pos = probe_world_position(probe_idx);

    // Accumulate irradiance into SH from ray samples.
    var sh_r_accum = vec4<f32>(0.0);
    var sh_g_accum = vec4<f32>(0.0);
    var sh_b_accum = vec4<f32>(0.0);
    var hit_count: u32 = 0u;

    let num_rays = min(volume.num_rays_per_probe, MAX_RAYS_PER_PROBE);
    for (var r: u32 = 0u; r < num_rays; r = r + 1u) {
        let ray_dir = normalize(ray_directions[r]);

        // March ray (simplified: single step at max distance for now).
        // Full implementation would trace against a depth representation.
        let hit_pos = probe_pos + ray_dir * volume.max_ray_distance;

        // Simplified irradiance: sample ambient + sky contribution at hit.
        // In production, this would trace against the scene and evaluate lighting.
        let sky_color = vec3<f32>(0.05, 0.05, 0.1); // placeholder ambient
        let ground_color = vec3<f32>(0.02, 0.03, 0.01);
        let irradiance = mix(ground_color, sky_color, max(ray_dir.y * 0.5 + 0.5, 0.0));

        let sh_update_r = project_sh(ray_dir, vec3<f32>(irradiance.r, 0.0, 0.0));
        let sh_update_g = project_sh(ray_dir, vec3<f32>(0.0, irradiance.g, 0.0));
        let sh_update_b = project_sh(ray_dir, vec3<f32>(0.0, 0.0, irradiance.b));

        sh_r_accum = sh_r_accum + sh_update_r;
        sh_g_accum = sh_g_accum + sh_update_g;
        sh_b_accum = sh_b_accum + sh_update_b;
        hit_count = hit_count + 1u;
    }

    if hit_count > 0u {
        let inv_hits = 1.0 / f32(hit_count);
        let weight = volume.energy_preservation * (4.0 * PI / f32(hit_count));

        // Weighted blend with existing coefficients.
        let alpha = 0.3; // hysteresis for temporal stability
        let old = probes[probe_idx];

        probes[probe_idx].sh_r = mix(old.sh_r, sh_r_accum * inv_hits * weight, alpha);
        probes[probe_idx].sh_g = mix(old.sh_g, sh_g_accum * inv_hits * weight, alpha);
        probes[probe_idx].sh_b = mix(old.sh_b, sh_b_accum * inv_hits * weight, alpha);
    }
}

// ── Entry point 2: Sample probes for indirect irradiance ──

/// Samples the probe volume to compute indirect irradiance per pixel.
/// Dispatched as 8x8 tiles covering the screen.
@compute @workgroup_size(8, 8, 1)
fn ddgi_sample_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    // Read world position and normal from G-buffer textures.
    let world_pos = textureLoad(world_position_texture, vec2<i32>(i32(px), i32(py)), 0).xyz;
    let world_normal = normalize(textureLoad(world_normal_texture, vec2<i32>(i32(px), i32(py)), 0).xyz);

    // Find the probe grid cell containing this world position.
    let local_pos = world_pos - volume.origin;
    let grid_f = local_pos / volume.spacing;

    let ix = u32(clamp(floor(grid_f.x), 0.0, f32(volume.extents.x - 1u)));
    let iy = u32(clamp(floor(grid_f.y), 0.0, f32(volume.extents.y - 1u)));
    let iz = u32(clamp(floor(grid_f.z), 0.0, f32(volume.extents.z - 1u)));

    // Fractional position within the grid cell.
    let frac = grid_f - vec3<f32>(f32(ix), f32(iy), f32(iz));
    let frac = clamp(frac, 0.0, 1.0);

    // Trilinear interpolation across 8 neighboring probes.
    var irradiance = vec3<f32>(0.0);
    var total_weight: f32 = 0.0;

    for (var dz: u32 = 0u; dz <= 1u; dz = dz + 1u) {
        for (var dy: u32 = 0u; dy <= 1u; dy = dy + 1u) {
            for (var dx: u32 = 0u; dx <= 1u; dx = dx + 1u) {
                let nx = min(ix + dx, volume.extents.x - 1u);
                let ny = min(iy + dy, volume.extents.y - 1u);
                let nz = min(iz + dz, volume.extents.z - 1u);

                let probe_idx = probe_index_3d_to_1d(nx, ny, nz);

                // Trilinear weight.
                let wx = mix(1.0 - frac.x, frac.x, f32(dx));
                let wy = mix(1.0 - frac.y, frac.y, f32(dy));
                let wz = mix(1.0 - frac.z, frac.z, f32(dz));
                let weight = wx * wy * wz;

                irradiance = irradiance + eval_sh_rgb(probes[probe_idx], world_normal) * weight;
                total_weight = total_weight + weight;
            }
        }
    }

    irradiance = irradiance / max(total_weight, 0.0001);

    // Write indirect irradiance to output texture.
    textureStore(irradiance_texture, vec2<i32>(i32(px), i32(py)), vec4<f32>(irradiance, 1.0));
}
