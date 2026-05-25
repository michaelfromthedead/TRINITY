# PHASE 3 ARCHITECTURE: Clustered Lighting

## Overview

Phase 3 implements GPU-side clustered lighting infrastructure. The CPU-side froxel grid and light culling (`light_culling.py`) are complete; this phase uploads culling results and enables GPU light iteration.

## Components

### 3.1 Froxel Grid GPU Buffer

The `FroxelGrid` class computes 3D frustum-aligned voxels with exponential depth slicing.

**Current State:**
- Grid dimensions computed (e.g., 16x9x24)
- Per-froxel AABB computed correctly
- No GPU upload

**Target State:**
- `FroxelBuffer`: Storage buffer with froxel bounds
- Uploaded once per frame (or on camera change)

**Buffer Layout:**
```wgsl
struct Froxel {
    aabb_min: vec3<f32>,
    _pad0: f32,
    aabb_max: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> froxels: array<Froxel>;
```

### 3.2 Light List Buffer

The `ClusteredLightCuller` produces per-froxel light lists.

**Current CPU Structure (`LightList`):**
```python
offset: int      # Start index in global light index array
count: int       # Number of lights in this froxel
```

**GPU Buffer Structure:**
```wgsl
struct LightListHeader {
    offset: u32,
    count: u32,
}

@group(0) @binding(1) var<storage, read> light_list_headers: array<LightListHeader>;
@group(0) @binding(2) var<storage, read> light_indices: array<u32>;
```

**Data Flow:**
```
CPU: ClusteredLightCuller.cull(lights, camera)
  -> Per-froxel light lists
  -> Flatten to (headers[], indices[])
  -> Upload to GPU buffers
```

### 3.3 Light Data Buffer

All lights need GPU-accessible data for evaluation.

**Per-Light Data:**
```wgsl
struct Light {
    position: vec3<f32>,
    light_type: u32,           // 0=dir, 1=point, 2=spot, 3=rect, 4=disk, 5=ies, 6=sky
    direction: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
    // Spot-specific
    inner_angle: f32,
    outer_angle: f32,
    // Area light-specific
    width: f32,
    height: f32,
    // Shadow-specific
    shadow_index: i32,         // -1 = no shadow, else index into shadow array
    _padding: vec3<f32>,
}
```

### 3.4 Light Evaluation Shader

Each light type has different attenuation math. The CPU implementation in `light_types.py` defines:

- **Point**: Inverse-square with radius-based smooth falloff
- **Spot**: Point attenuation * angular attenuation (smoothstep)
- **Rect/Disk**: LTC-based (requires LUT texture)
- **IES**: Profile lookup with bilinear sampling

**WGSL Light Evaluation:**
```wgsl
fn evaluate_light(light: Light, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    switch (light.light_type) {
        case 0u: { return evaluate_directional(light, normal); }
        case 1u: { return evaluate_point(light, world_pos, normal); }
        case 2u: { return evaluate_spot(light, world_pos, normal); }
        // ...
    }
}

fn evaluate_point(light: Light, pos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let L = light.position - pos;
    let dist = length(L);
    if (dist > light.radius) { return vec3(0.0); }
    let L_norm = L / dist;
    let NdotL = max(dot(N, L_norm), 0.0);
    let attenuation = smooth_distance_attenuation(dist, light.radius);
    return light.color * light.intensity * NdotL * attenuation;
}

fn smooth_distance_attenuation(dist: f32, radius: f32) -> f32 {
    let ratio = clamp(dist / radius, 0.0, 1.0);
    let ratio2 = ratio * ratio;
    let factor = 1.0 - ratio2 * ratio2;
    return factor * factor / max(dist * dist, 0.0001);
}
```

### 3.5 Clustered Lighting Loop

The main lighting shader iterates lights per froxel:

```wgsl
fn shade_fragment(world_pos: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let froxel_idx = compute_froxel_index(world_pos);
    let header = light_list_headers[froxel_idx];
    
    var radiance = vec3(0.0);
    for (var i = 0u; i < header.count; i++) {
        let light_idx = light_indices[header.offset + i];
        let light = lights[light_idx];
        radiance += evaluate_light(light, world_pos, normal);
    }
    return radiance;
}
```

## Architecture Decisions

### Why Storage Buffers Over Textures?
- Arbitrary-length light lists (not limited to texture dimensions)
- Easier CPU-side packing
- Modern GPU support (WebGPU requires storage buffers)

### Why Not Compute Culling?
The CPU culling is sufficient for:
- Light counts < 10,000 (CPU handles fine)
- Static or slowly-moving lights
- Predictable frame timing

GPU culling would be needed for:
- Light counts > 10,000
- Highly dynamic light sources
- Compute-shader-based rendering pipeline

Decision: **Keep CPU culling for Phase 3**, add GPU culling in future optimization phase.

## Data Flow Summary

```
Frame Start
  |
  v
CPU: Update light positions/properties
  |
  v
CPU: FroxelGrid.update(camera)  [if camera changed]
  |
  v
CPU: ClusteredLightCuller.cull(lights, camera)
  |
  v
CPU: Pack light lists + light data into buffers
  |
  v
GPU: Upload light_list_headers, light_indices, lights
  |
  v
GPU: Fragment shader uses clustered lighting loop
```

## Buffer Sizing

| Buffer | Formula | Example (1000 lights, 16x9x24 grid) |
|--------|---------|-------------------------------------|
| Froxels | 32 bytes * X * Y * Z | 32 * 16 * 9 * 24 = 110 KB |
| Light Headers | 8 bytes * X * Y * Z | 8 * 3456 = 27 KB |
| Light Indices | 4 bytes * (avg lights/froxel * froxels) | 4 * 10 * 3456 = 138 KB |
| Lights | 64 bytes * light_count | 64 * 1000 = 64 KB |

Total: ~340 KB for 1000 lights. Well within GPU limits.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Light list overflow | Dynamic reallocation or conservative max |
| Froxel grid mismatch | Sync grid params between CPU and shader constants |
| Depth slice precision | Use conservative depth bounds per froxel |
| Many lights per froxel | Cap per-froxel count, sort by importance |
