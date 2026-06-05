# PHASE 4 ARCHITECTURE: Global Illumination

## Overview

Phase 4 implements GPU-side global illumination, including spherical harmonics probes, DDGI (Dynamic Diffuse Global Illumination), lightmaps, and reflection probes. The CPU algorithms in `gi_probes.py` and `gi_ddgi.py` are complete; this phase creates GPU textures and shaders.

## Components

### 4.1 Spherical Harmonics Probes

**Current State (gi_probes.py:69-95):**
- L2 SH with 27 coefficients (9 per RGB channel)
- `evaluate(direction)` returns irradiance
- `add_sample(direction, color)` accumulates radiance
- Probes arranged in 3D grid with trilinear interpolation

**GPU Representation:**
```wgsl
struct SHProbe {
    coefficients: array<vec4<f32>, 7>,  // 27 floats packed into 7 vec4s (28 slots, 1 unused)
}
```

**Storage Options:**
1. **3D Texture**: 3x SH textures for R, G, B (9 texels each)
2. **Storage Buffer**: Array of SHProbe structs
3. **Texture Array**: 2D slices for probe grid layers

Decision: **Storage buffer** for flexibility; probes are sparse in most scenes.

### 4.2 SH Evaluation Shader

**CPU Reference (gi_probes.py:74-95):**
```python
y0 = 0.282095  # 1/(2*sqrt(pi))
y1 = 0.488603 * d.y
y6 = 0.315392 * (3 * d.z * d.z - 1)
```

**WGSL Translation:**
```wgsl
fn evaluate_sh_l2(probe: SHProbe, direction: vec3<f32>) -> vec3<f32> {
    let d = normalize(direction);
    
    // L0 band
    let y0 = 0.282095;
    
    // L1 band
    let y1 = 0.488603 * d.y;
    let y2 = 0.488603 * d.z;
    let y3 = 0.488603 * d.x;
    
    // L2 band
    let y4 = 1.092548 * d.x * d.y;
    let y5 = 1.092548 * d.y * d.z;
    let y6 = 0.315392 * (3.0 * d.z * d.z - 1.0);
    let y7 = 1.092548 * d.x * d.z;
    let y8 = 0.546274 * (d.x * d.x - d.y * d.y);
    
    // Dot product with coefficients
    // ... unpack and accumulate
}
```

### 4.3 Probe Grid Interpolation

3D trilinear interpolation between 8 nearest probes:

```wgsl
fn sample_probe_grid(world_pos: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let local = (world_pos - grid_origin) / grid_spacing;
    let cell = floor(local);
    let frac = fract(local);
    
    // Sample 8 corner probes
    var irradiance = vec3(0.0);
    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let idx = get_probe_index(cell + vec3(f32(x), f32(y), f32(z)));
                let weight = trilinear_weight(frac, vec3(f32(x), f32(y), f32(z)));
                irradiance += weight * evaluate_sh_l2(probes[idx], direction);
            }
        }
    }
    return irradiance;
}
```

### 4.4 DDGI GPU Textures

**Current State (gi_ddgi.py):**
- Octahedral encoding of irradiance and visibility
- Per-probe ray tracing with Fibonacci spiral
- Chebyshev visibility test for soft indirect shadows

**GPU Texture Layout:**
```
Irradiance Texture (RGBA16F):
  - Each probe: 8x8 texels (octahedral)
  - Probes tiled in 2D atlas
  - RGB = irradiance, A = unused

Visibility Texture (RG16F):
  - Each probe: 16x16 texels (higher res for sharp visibility)
  - R = mean distance, G = mean distance^2
  - Chebyshev test uses both moments
```

### 4.5 Octahedral Encoding

**CPU Reference (gi_ddgi.py:131-153):**
```python
def _direction_to_octahedral(self, direction: Vec3) -> Vec2:
    inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z))
    ox = d.x * inv_l1
    oy = d.y * inv_l1
    if d.z < 0:
        ox = (1.0 - abs(oy)) * (1.0 if ox >= 0 else -1.0)
        oy = (1.0 - abs(ox)) * (1.0 if oy >= 0 else -1.0)
```

**WGSL Translation:**
```wgsl
fn direction_to_octahedral(direction: vec3<f32>) -> vec2<f32> {
    let d = normalize(direction);
    let inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z));
    var o = d.xy * inv_l1;
    if (d.z < 0.0) {
        o = (1.0 - abs(o.yx)) * sign(o);
    }
    return o * 0.5 + 0.5;  // [0, 1] range
}

fn octahedral_to_direction(oct: vec2<f32>) -> vec3<f32> {
    let o = oct * 2.0 - 1.0;
    var d = vec3(o.xy, 1.0 - abs(o.x) - abs(o.y));
    if (d.z < 0.0) {
        d = vec3((1.0 - abs(d.yx)) * sign(d.xy), d.z);
    }
    return normalize(d);
}
```

### 4.6 DDGI Update Pass

Ray tracing pass updates probe textures:

```
For each probe in update set:
    For each ray in Fibonacci spiral:
        Trace ray from probe position
        On hit: accumulate irradiance from direct lighting
        Record hit distance for visibility
    
    Write to irradiance texture (octahedral)
    Write to visibility texture (moments)
```

**Integration Options:**
1. **Hardware RT**: `rayQueryEXT` in compute shader (requires RTX/DXR)
2. **Software RT**: BVH traversal in compute shader
3. **Callback**: Existing `trace_func` parameter accepts external RT system

Decision: **Callback approach** initially; add native RT later.

### 4.7 DDGI Lookup with Visibility

**CPU Reference (gi_ddgi.py - Chebyshev test):**
```python
variance = max(moments.y - moments.x * moments.x, 0.0001)
visibility = variance / (variance + (distance - moments.x)**2)
```

**WGSL:**
```wgsl
fn ddgi_sample(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Find enclosing probes
    // For each probe:
    let oct_uv = direction_to_octahedral(normal);
    let irradiance = textureSample(ddgi_irradiance, sampler, probe_uv(probe_idx, oct_uv)).rgb;
    let moments = textureSample(ddgi_visibility, sampler, probe_uv(probe_idx, oct_uv)).rg;
    
    // Chebyshev visibility
    let dist_to_probe = length(world_pos - probe_position);
    let variance = max(moments.y - moments.x * moments.x, 0.0001);
    let visibility = variance / (variance + pow(max(dist_to_probe - moments.x, 0.0), 2.0));
    
    // Weight by visibility and trilinear
    return irradiance * visibility * trilinear_weight;
}
```

### 4.8 Reflection Probes

**Current State (gi_probes.py:665-670):**
```python
def sample(self, world_pos, reflection_dir, roughness=0.0):
    corrected_dir = self._parallax_correct(world_pos, reflection_dir)
    return Vec3(0.5, 0.5, 0.5)  # Placeholder
```

**GPU Implementation:**
- Cubemap array texture
- Parallax box correction per probe
- Roughness-based mip selection
- Blending between overlapping probes

```wgsl
fn sample_reflection_probe(probe: ReflectionProbe, world_pos: vec3<f32>, 
                           reflection: vec3<f32>, roughness: f32) -> vec3<f32> {
    let corrected = parallax_correct(probe, world_pos, reflection);
    let mip = roughness_to_mip(roughness, probe.mip_count);
    return textureSampleLevel(cubemap_array, sampler, corrected, probe.index, mip).rgb;
}
```

### 4.9 Lightmaps

Baked per-texel irradiance:

**Storage:** 2D texture per lightmapped object (or atlas)
**Sampling:** UV2 (lightmap UV) lookup with bilinear filtering
**Integration:** Added to indirect lighting term

## Data Flow

```
Bake Time:
  SH Probes: Fibonacci spiral sampling -> SH coefficients -> buffer
  Lightmaps: Path tracing -> texels -> texture
  Reflection Probes: Cubemap capture -> texture array

Runtime (per frame):
  DDGI Update Pass: Trace rays -> irradiance/visibility textures
  Lighting Pass: Sample all GI sources -> accumulate indirect
```

## Memory Budget

| Resource | Formula | Example |
|----------|---------|---------|
| SH Probes | 112 bytes * probe_count | 112 * 1000 = 112 KB |
| DDGI Irradiance | 8x8 * 4 bytes * probe_count | 256 * 1000 = 256 KB |
| DDGI Visibility | 16x16 * 4 bytes * probe_count | 1024 * 1000 = 1 MB |
| Reflection Cubemaps | 6 * 128^2 * 4 * mips * count | ~2 MB per probe |
| Lightmaps | texture_size * 4 bytes | Variable, ~10 MB typical |

Total: 10-50 MB depending on scene complexity.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| DDGI update cost | Prioritized updates (N probes/frame), temporal spread |
| Reflection probe pop-in | Blend factor over multiple frames |
| SH ringing | Windowed SH or smooth clamping |
| Lightmap UV artifacts | Padding between charts, bilinear clamp |
