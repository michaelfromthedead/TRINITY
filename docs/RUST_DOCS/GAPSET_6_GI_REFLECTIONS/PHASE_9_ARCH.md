# Phase 9: Denoising Infrastructure -- Architecture

## Overview

Phase 9 implements the denoising pipeline that supports DDGI temporal accumulation, SSGI temporal, SSR temporal, and RT reflection denoising: A-trous wavelet spatial denoiser, temporal denoiser, and SVGF-style variance estimation.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P9.1 | [-] | Implement A-trous wavelet spatial denoiser |
| T-GIR-P9.2 | [-] | Implement temporal denoiser |
| T-GIR-P9.3 | [-] | Implement SVGF-style variance estimation |

## Current State: NOT BUILT

No denoising code exists anywhere in the codebase.

## Required Architecture

### A-trous Wavelet Spatial Denoiser (T-GIR-P9.1)

Design per Dammertz et al. "Edge-Avoiding A-Trous Wavelet Transform" (2010):

```
Input: Noisy buffer (e.g., 1-spp GI or reflections)
Output: Spatially denoised buffer

For each iteration i = 0..4:
  step_size = 2^i  (1, 2, 4, 8, 16)
  
  For each pixel:
    - Sample center and 4+4 neighbor pixels along +X, -X, +Y, -Y at step_size distance
    - Weight each neighbor by:
      a) Gaussian spatial weight (constant for given step_size)
      b) Edge-stopping weight from normal deviation
      c) Edge-stopping weight from depth difference
      d) Edge-stopping weight from luminance difference
    
    - Apply bilateral filter: result = sum(neighbor * weight) / sum(weight)
    - Edge-stopping functions use sigma parameters:
      * normal_sigma: typically 0.1-0.5 (cosine angle threshold)
      * depth_sigma: typically 1.0-10.0 (depth delta in world units)
      * luminance_sigma: typically 0.1-0.5 (relative luminance)
```

**Edge-stopping functions** (`denoise_edge_stop.wgsl`):
```
fn edge_stop_normal(n1: vec3<f32>, n2: vec3<f32>, sigma: f32) -> f32 {
    return exp(-max(0.0, 1.0 - dot(n1, n2)) / sigma);
}

fn edge_stop_depth(d1: f32, d2: f32, sigma: f32) -> f32 {
    return exp(-abs(d1 - d2) / (sigma * max(d1, 0.01)));
}

fn edge_stop_luminance(l1: f32, l2: f32, sigma: f32) -> f32 {
    return exp(-abs(l1 - l2) / sigma);
}
```

### Temporal Denoiser (T-GIR-P9.2)

```
Input: Current frame denoised (from A-trous), previous frame result, velocity buffer
Output: Temporally accumulated result

For each pixel:
  1. Reproject: previous_uv = current_uv + velocity(current_uv)
  2. Validate reprojection: check if previous_uv is within screen bounds
  3. Neighborhood clamp:
     - Compute 3x3 neighborhood mean and variance in current frame
     - Clamp history sample to [mean - sigma, mean + sigma] (AABB clamp)
  4. Blend: result = lerp(current, clamped_history, feedback_factor)
  5. feedback_factor is typically 0.8-0.95:
     - Higher = more stable (static scenes)
     - Lower = less ghosting (dynamic scenes)
  6. Disocclusion handling:
     - If velocity is invalid (first frame, disocclusion): reset history
     - Use depth-based disocclusion threshold
```

### SVGF Variance Estimation (T-GIR-P9.3)

Spatiotemporal Variance-Guided Filtering (Schied et al. 2017):

```
1. Variance estimation:
   - Use temporal accumulation to estimate mean and variance per pixel
   - variance = E[x^2] - E[x]^2 (accumulated over frames)
   - Temporal variance estimation: lerp(prev_variance, current_variance, alpha)

2. Adaptive filter width:
   - High variance -> wider filter (more aggressive denoising)
   - Low variance -> narrower filter (preserve detail)
   - Filter width derived from variance-to-mean ratio

3. Second A-trous pass with variance-guided parameters:
   - Use variance to set edge-stop sigmas
   - Higher variance -> wider spatial filter

4. Temporal accumulation with variance boostrap:
   - Use filtered result as better estimate for temporal accumulation
   - Iterative refinement over frames
```

## Dependencies

- T-GIR-P9.1: S8 (velocity buffer for edge-stopping in temporal domain)
- T-GIR-P9.2: T-GIR-P9.1, S8 (velocity buffer)
- T-GIR-P9.3: T-GIR-P9.1, T-GIR-P9.2

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/denoise_atrous.comp.wgsl` | A-trous wavelet spatial denoiser |
| `crates/renderer-backend/shaders/denoise_edge_stop.wgsl` | Edge-stopping functions |
| `crates/renderer-backend/shaders/denoise_temporal.comp.wgsl` | Temporal denoiser |
| `crates/renderer-backend/shaders/denoise_svgf.comp.wgsl` | SVGF variance estimation |
| `crates/renderer-backend/src/denoise_state.rs` | Denoiser state, ping-pong buffers, history management |

## Integration Points

The denoising pipeline feeds into:

| Consumer | What It Denoises |
|----------|-----------------|
| Phase 2 (DDGI) | DDGI temporal probe update |
| Phase 3 (SSGI) | SSGI temporal accumulation |
| Phase 4 (SSR) | SSR temporal reprojection |
| Phase 8 (RT) | RT reflection denoising |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| A-trous denoiser removes noise while preserving edges | Failing -- not built |
| Temporal denoiser stabilizes over 8-16 frames | Failing -- not built |
| SVGF adapts filter width to local variance | Failing -- not built |
| Denoiser handles disocclusion without ghosting | Failing -- not built |
| Full denoising pass <4ms at 1080p | Failing -- not built |
