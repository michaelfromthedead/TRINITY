# Phase 8: RT Reflections -- Architecture

## Overview

Phase 8 implements hardware-accelerated ray-traced reflections: ray generation with BRDF importance sampling, roughness-based ray count adaptation, denoising, and a fallback chain for when RT is unavailable or inappropriate.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P8.1 | [-] | Implement RT reflection ray generation |
| T-GIR-P8.2 | [-] | Implement BRDF importance sampling for RT reflections |
| T-GIR-P8.3 | [-] | Implement roughness-based ray count adaptation |
| T-GIR-P8.4 | [-] | Implement RT reflection denoising |
| T-GIR-P8.5 | [-] | Implement reflection fallback chain |

## Current State: Infrastructure Only

### Frame Graph Support (Existing)

The frame graph IR (`crates/renderer-backend/src/frame_graph/mod.rs`) supports RT passes:

```rust
pub enum PassType {
    // ...
    RayTracing,
    // ...
}

pub enum ViewType {
    // ...
    AccelerationStructure,
    // ...
}

impl IrPass {
    pub fn ray_tracing(name: &str) -> Self { ... }
}
```

These are **infrastructure-only**: the pass type exists but no TLAS management, SBT construction, or RT shader dispatch is implemented.

### PBR BRDF (Existing in `pbr.frag.wgsl`)

Cook-Torrance BRDF with GGX:
- `distribution_ggx(n, h, roughness)` -- Trowbridge-Reitz NDF
- `geometry_smith(n, v, l, roughness)` -- Smith-GGX geometry
- `fresnel_schlick(cos_theta, f0)` -- Schlick Fresnel

The BRDF exists but only for direct lighting. No importance sampling for RT exists.

## Required Architecture

### RT Ray Generation (T-GIR-P8.1)

```
Required infrastructure (S10):
1. TLAS (Top-Level Acceleration Structure): scene instance transforms + BLAS references
2. BLAS (Bottom-Level Acceleration Structures): per-mesh geometry
3. SBT (Shader Binding Table): ray generation, hit, miss shader records

RT Reflection shader pipeline:
1. Ray Generation Shader (rt_reflections.rgen):
   - Read G-buffer (position, normal, roughness, metallic)
   - Generate reflection ray direction (see T-GIR-P8.2)
   - Trace ray against TLAS
   - Write hit result to reflection buffer

2. Closest-Hit Shader (rt_reflections.rchit):
   - Fetch material at hit point
   - Compute direct lighting at hit point
   - Return accumulated radiance

3. Miss Shader (rt_reflections.rmiss):
   - Return sky/environment radiance
```

### BRDF Importance Sampling (T-GIR-P8.2)

Generate reflection ray directions weighted by the Cook-Torrance BRDF:

```
1. Sample GGX distribution for microfacet normal:
   - theta = atan(a * sqrt(xi1) / sqrt(1 - xi1)), where a = roughness^2
   - phi = 2 * PI * xi2

2. Compute reflection direction:
   - half_vector = GGX sample
   - reflection = reflect(view, half_vector)

3. PDF: D(h) * dot(n, h) / (4 * dot(v, h))
   - Used for multiple-importance sampling (MIS) weight
```

### Roughness Adaptation (T-GIR-P8.3)

```
1. Classify pixels by roughness:
   - Smooth (roughness < 0.2): 4-8 rays/pixel, full resolution
   - Medium (0.2 < roughness < 0.6): 2-4 rays/pixel, half resolution
   - Rough (roughness > 0.6): 1 ray/pixel, quarter resolution

2. Resolution hierarchy:
   - Full res for smooth surfaces (sharp reflections need detail)
   - Half/quarter res for rough surfaces (blurred reflections)

3. Temporal accumulation substitutes ray count:
   - Fewer rays per frame, accumulate over multiple frames
   - Use velocity buffer for reprojection
```

### RT Denoising (T-GIR-P8.4)

```
Input: Noisy RT reflection buffer (1 spp), G-buffer (normal, depth, roughness)
Output: Denoised reflection buffer

Two-stage denoiser:
1. Spatial denoising (A-trous wavelet):
   - Edge-stopping functions based on normal deviation, depth delta, luminance
   - 4-5 iterations with increasing kernel step size (1, 2, 4, 8, 16)

2. Temporal denoising:
   - Reproject previous denoised frame
   - Neighborhood clamping
   - Exponential moving average blend
```

### Reflection Fallback Chain (T-GIR-P8.5)

Per-pixel fallback strategy:

```
1. If RT available AND smooth surface: use RT reflections (T-GIR-P8.1)
2. If RT available AND rough surface: use RT reflections at reduced rate (T-GIR-P8.3)
3. If RT not available AND screen-space hit: use SSR (T-GIR-P4.2)
4. If SSR miss: use reflection probes (T-GIR-P5.3)
5. If probe miss: use environment map / sky

Decision encoded in per-pixel technique mask buffer.
```

## Dependencies

- T-GIR-P8.1: S10 (TLAS, SBT infrastructure)
- T-GIR-P8.2: T-GIR-P8.1, PBR BRDF
- T-GIR-P8.3: T-GIR-P8.1
- T-GIR-P8.4: T-GIR-P8.1, S8 (velocity buffer), T-GIR-P9.1 (A-trous), T-GIR-P9.2 (temporal denoiser)
- T-GIR-P8.5: T-GIR-P4.2, T-GIR-P5.3, T-GIR-P8.1

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/rt_reflections.rgen` | RT reflection ray generation |
| `crates/renderer-backend/shaders/rt_reflections.rchit` | RT reflection closest-hit |
| `crates/renderer-backend/shaders/rt_reflections.rmiss` | RT reflection miss |
| `crates/renderer-backend/shaders/rt_reflections_denoise.comp.wgsl` | RT denoising |
| `crates/renderer-backend/shaders/reflection_fallback_chain.comp.wgsl` | Per-pixel fallback |
| `crates/renderer-backend/src/rt_reflections.rs` | RT reflection pass management |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| RT reflections produce correct mirror reflections | Failing -- not built |
| BRDF sampling converges to correct integral (MIS weight) | Failing -- not built |
| Roughness adaptation reduces cost for rough surfaces | Failing -- not built |
| Denoiser converges to stable result within 8 frames | Failing -- not built |
| Fallback chain handles all reflection scenarios | Failing -- not built |
| RT reflections run at <8ms at 1080p (with denoising) | Failing -- not built |
