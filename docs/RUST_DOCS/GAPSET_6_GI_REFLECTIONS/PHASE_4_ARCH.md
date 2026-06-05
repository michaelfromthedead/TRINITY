# Phase 4: SSR Core -- Screen-Space Reflections -- Architecture

## Overview

Phase 4 implements screen-space reflections: HiZ buffer generation, HiZ-accelerated ray marching, linear ray marching fallback, temporal reprojection, and roughness-driven blur (Bloomberg multi-downsample).

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P4.1 | [-] | Implement HiZ buffer generation |
| T-GIR-P4.2 | [-] | Implement SSR HiZ ray marching |
| T-GIR-P4.3 | [-] | Implement SSR linear ray marching fallback |
| T-GIR-P4.4 | [-] | Implement SSR temporal reprojection |
| T-GIR-P4.5 | [-] | Implement SSR roughness-driven blur |

## Current State: NOT BUILT

No SSR code exists anywhere in the codebase. No HiZ buffer generation. No SSR shaders. The PBR fragment shader (`pbr.frag.wgsl`) handles direct lighting only -- no reflection integration.

## Required Architecture

### HiZ Buffer Generation (T-GIR-P4.1)

Hierarchical Z-buffer: a mip chain of the depth buffer where each level stores the maximum (or minimum) depth of the 2x2 block above.

```
Input: Full-res depth buffer (texture_depth_2d)
Output: HZB mip chain (up to ~10 levels at 1080p)

Level 0: Full resolution depth
Level 1: 2x2 max reduction -> half resolution
Level 2: 2x2 max reduction -> quarter resolution
...continue until 1x1 or 2x2
```

Dispatch: compute shader with workgroup_size(8, 8, 1), each workgroup processes a 16x16 tile with shared memory.

### SSR HiZ Ray Marching (T-GIR-P4.2)

Ray march against the HZB for acceleration (per Yasin Uludag, "Hi-Z Screen-Space Reflections" in GPU Pro 5):

```
Input: G-buffer (position, normal, roughness), HZB mip chain, reflected ray direction
Output: Reflection color at intersection point

For each pixel:
  1. Compute reflected ray from view direction and surface normal
  2. Start at surface position, step along reflected ray in UV space
  3. At each step, query HZB at appropriate mip level (level = floor(log2(stride)))
  4. Compare ray depth against HZB depth at that level
  5. If ray depth > HZB depth: refine with finer mip levels (binary search)
  6. If intersection found: sample G-buffer albedo at hit UV
  7. Apply fade based on: screen edge distance, ray length, roughness
```

### SSR Linear Marching Fallback (T-GIR-P4.3)

Simpler ray march using fixed step size (no HZB acceleration):

```
Input: Same as HiZ path
Output: Reflection color

- Use when HZB is unavailable or at very low resolutions
- Fixed step count (e.g., 32-64 steps)
- Fade out based on step count before termination
- Higher performance, lower quality
```

### SSR Temporal Reprojection (T-GIR-P4.4)

```
Input: Current SSR, previous SSR result, velocity buffer
Output: Temporally accumulated SSR

1. Reproject using velocity buffer
2. Neighborhood clamp (3x3 variance-based AABB)
3. History blend with exponential moving average
4. Disocclusion detection: reset history when velocity > threshold
```

### SSR Roughness-Driven Blur (T-GIR-P4.5)

Bloomberg multi-downsample approach (from UE4/SIGGRAPH 2014):

```
Input: SSR result buffer, roughness buffer
Output: Blurred SSR per roughness level

1. Downsample SSR into 4-5 mip levels (2x2 box filter)
2. At each level, apply separable Gaussian blur with kernel size proportional to roughness
3. Select mip level based on surface roughness (smoother = higher mip)
4. Blend between mip levels for continuous roughness response
```

## Dependencies

- T-GIR-P4.1: S1 (frame graph dispatch)
- T-GIR-P4.2: T-GIR-P4.1
- T-GIR-P4.3: T-GIR-P4.1
- T-GIR-P4.4: T-GIR-P4.2, S8 (velocity buffer)
- T-GIR-P4.5: T-GIR-P4.4

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/hiz_generate.comp.wgsl` | HZB mip chain generation |
| `crates/renderer-backend/shaders/ssr_ray_march.comp.wgsl` | SSR HiZ ray marching |
| `crates/renderer-backend/shaders/ssr_ray_march_linear.comp.wgsl` | SSR linear fallback |
| `crates/renderer-backend/shaders/ssr_temporal.comp.wgsl` | SSR temporal reprojection |
| `crates/renderer-backend/shaders/ssr_blur.comp.wgsl` | SSR roughness blur |
| `crates/renderer-backend/src/ssr.rs` | SSR pass management |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| HiZ buffer generates correct depth mip chain | Failing -- not built |
| SSR produces plausible reflections on smooth surfaces | Failing -- not built |
| SSR temporal accumulation stabilizes reflections | Failing -- not built |
| SSR roughness blur matches BRDF lobe width | Failing -- not built |
| SSR runs at <5ms at 1080p | Failing -- not built |
