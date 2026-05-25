# Phase 3: SSGI -- Screen-Space GI -- Architecture

## Overview

Phase 3 implements screen-space global illumination: ray marching against the depth buffer to compute indirect diffuse lighting, followed by temporal accumulation for stability.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P3.1 | [-] | Implement SSGI ray marching |
| T-GIR-P3.2 | [-] | Implement SSGI temporal accumulation |

## Current State: NOT BUILT

No SSGI code exists anywhere in the codebase. The plan describes a complete implementation -- no source files match the described components.

## Required Architecture

### SSGI Ray March (T-GIR-P3.1)

Design per UE4 SSGI and Crytek SSAO techniques:

```
Input: G-buffer (world position, world normal), depth buffer HZB mip chain
Output: Indirect diffuse irradiance buffer (half-res RGBA16F)

For each pixel:
  1. Generate N cosine-weighted hemisphere samples oriented by surface normal
  2. For each sample:
     a. Project 3D sample direction to screen-space UV
     b. March along ray using HZB (hierarchical depth buffer) for acceleration
     c. Find hit point where ray depth crosses scene depth
     d. Accumulate indirect diffuse from hit point's albedo
  3. Average over N samples
```

### SSGI Temporal Accumulation (T-GIR-P3.2)

```
Input: Current frame SSGI, previous frame SSGI, velocity buffer
Output: Temporally accumulated SSGI

For each pixel:
  1. Reproject previous frame SSGI using velocity buffer
  2. Clamp history to current frame's neighborhood (AABB clamp)
  3. Blend: result = lerp(current, history, feedback_factor)
  4. Detect disocclusion: reset history where velocity is invalid
```

## Dependencies

- T-GIR-P3.1: T-GIR-P4.1 (HiZ buffer generation -- also needed by SSR)
- T-GIR-P3.2: T-GIR-P3.1, S8 (velocity buffer)

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/ssgi_trace.comp.wgsl` | SSGI ray marching compute shader |
| `crates/renderer-backend/shaders/ssgi_temporal.comp.wgsl` | SSGI temporal accumulation |
| `crates/renderer-backend/src/ssgi.rs` | SSGI pass creation and state management |
| `engine/rendering/lighting/gi_ssgi.py` | Python reference if desired |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| SSGI ray march produces plausible indirect lighting | Failing -- not built |
| SSGI temporal accumulation stabilizes over frames | Failing -- not built |
| SSGI runs at <3ms at 1080p | Failing -- not built |
