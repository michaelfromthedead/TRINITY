# PHASE 2 ARCHITECTURE: Shadow Filtering Shaders

## Overview

Phase 2 implements WGSL shaders for all shadow filtering techniques. The CPU code in `shadow_filtering.py` defines the algorithms; this phase translates them to GPU-executable shaders.

## Components

### 2.1 PCF Shader Variants

The `PCFFilter` class supports three sampling patterns:
- **Grid**: Regular NxN samples
- **Poisson disk**: Irregular distribution for less banding
- **Vogel disk**: Spiral pattern for smooth falloff

Each pattern becomes a shader variant or compile-time constant.

**WGSL Structure:**
```wgsl
fn pcf_sample_grid(shadow_map: texture_depth_2d, sampler: sampler_comparison, 
                   uv: vec2<f32>, depth: f32, filter_size: f32, samples: u32) -> f32 {
    var visibility = 0.0;
    let step = filter_size / f32(samples);
    for (var y = 0u; y < samples; y++) {
        for (var x = 0u; x < samples; x++) {
            let offset = vec2<f32>(f32(x) - f32(samples)/2.0, f32(y) - f32(samples)/2.0) * step;
            visibility += textureSampleCompare(shadow_map, sampler, uv + offset, depth);
        }
    }
    return visibility / f32(samples * samples);
}
```

### 2.2 PCSS Shader

PCSS requires two passes:
1. **Blocker search**: Find average depth of occluders
2. **Filtering**: Variable-width PCF based on blocker distance

**Current CPU Code (shadow_filtering.py:428-446):**
```python
def _estimate_penumbra(self, receiver_depth: float, blocker_depth: float) -> float:
    return (receiver_depth - blocker_depth) / blocker_depth
```

**WGSL Translation:**
```wgsl
fn estimate_penumbra(receiver_depth: f32, blocker_depth: f32, light_size: f32) -> f32 {
    if (blocker_depth <= 0.0 || blocker_depth >= receiver_depth) {
        return 0.0;
    }
    return light_size * (receiver_depth - blocker_depth) / blocker_depth;
}
```

### 2.3 VSM Shader

Variance Shadow Maps use Chebyshev's inequality.

**Current CPU Code:**
- Stores `(depth, depth^2)` per texel
- Uses Chebyshev test for soft shadows
- Light bleeding reduction via clamping

**WGSL Implementation:**
```wgsl
fn vsm_visibility(moments: vec2<f32>, depth: f32, min_variance: f32, bleed_reduction: f32) -> f32 {
    let p = step(depth, moments.x);  // Hard shadow term
    let variance = max(moments.y - moments.x * moments.x, min_variance);
    let d = depth - moments.x;
    let p_max = variance / (variance + d * d);
    let reduced = clamp((p_max - bleed_reduction) / (1.0 - bleed_reduction), 0.0, 1.0);
    return max(p, reduced);
}
```

### 2.4 ESM Shader

Exponential Shadow Maps use exponential depth warping.

**WGSL Implementation:**
```wgsl
fn esm_visibility(exp_depth: f32, receiver_depth: f32, exponent: f32) -> f32 {
    return clamp(exp(-exponent * receiver_depth) * exp_depth, 0.0, 1.0);
}
```

### 2.5 Contact Shadows

Screen-space ray march for near-contact detail.

**Current CPU Code (shadow_filtering.py:703-704):**
```python
# In production, this would:
# 1. Ray march from shading point toward light in screen space
# Placeholder return
return ShadowSample(visibility=1.0)
```

**WGSL Implementation:**
```wgsl
fn contact_shadow(depth_buffer: texture_2d<f32>, screen_pos: vec2<f32>, 
                  light_dir_ss: vec2<f32>, max_steps: u32, thickness: f32) -> f32 {
    var pos = screen_pos;
    for (var i = 0u; i < max_steps; i++) {
        pos += light_dir_ss;
        let sampled_depth = textureSample(depth_buffer, sampler, pos).r;
        let current_depth = linearize_depth(screen_pos, i);
        if (sampled_depth < current_depth && current_depth - sampled_depth < thickness) {
            return 0.0;  // Occluded
        }
    }
    return 1.0;  // Visible
}
```

## Architecture Decision: Shader Permutations

Rather than runtime branching, generate shader permutations:

| Permutation Key | Values | Shader Constant |
|-----------------|--------|-----------------|
| `FILTER_TYPE` | PCF, PCSS, VSM, ESM | Compile-time select |
| `PCF_PATTERN` | GRID, POISSON, VOGEL | Compile-time select |
| `SHADOW_TYPE` | DIRECTIONAL, SPOT, POINT | Affects sampling |
| `USE_CONTACT` | 0, 1 | Enable contact pass |

Total permutations: 4 * 3 * 3 * 2 = 72 (but many combinations are invalid, real count ~20)

## Data Flow

```
Fragment Shader
  -> Read shadow map UV from light data
  -> Transform world position to shadow space
  -> Call selected filter function
  -> Combine with contact shadow if enabled
  -> Output visibility [0,1]
```

## Uniform Buffers

### ShadowFilterUniforms
```wgsl
struct ShadowFilterUniforms {
    filter_size: f32,           // PCF kernel size
    pcss_light_size: f32,       // For penumbra estimation
    vsm_min_variance: f32,      // Anti-bleeding
    vsm_bleed_reduction: f32,   // Light bleed fix
    esm_exponent: f32,          // Depth warping strength
    contact_max_steps: u32,     // Ray march iterations
    contact_thickness: f32,     // Occlusion thickness
    _padding: f32,
}
```

## Integration Points

- **Phase 1 output**: Shadow map textures and samplers
- **Light culling**: Per-light shadow data (view-proj, atlas region)
- **Main lighting shader**: Calls filter functions

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Permutation explosion | Only compile used combinations on demand |
| PCSS blocker search cost | Configurable search radius, skip for distant shadows |
| ESM precision loss | Use R32Float, clamp exponent to 40-80 |
| Contact shadow artifacts | Jitter ray start, temporal accumulation |
