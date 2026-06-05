# Phase 7: Voxel GI -- Architecture

## Overview

Phase 7 implements voxel-based global illumination: scene voxelisation, voxel mip chain generation and storage, and voxel cone tracing for indirect lighting.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P7.1 | [-] | Implement scene voxelisation |
| T-GIR-P7.2 | [-] | Implement voxel mip chain and storage |
| T-GIR-P7.3 | [-] | Implement voxel cone tracing |

## Current State: NOT BUILT

No voxel GI code exists anywhere in the codebase.

## Required Architecture

### Scene Voxelisation (T-GIR-P7.1)

Design per Crassin et al. "Interactive Indirect Illumination Using Voxel Cone Tracing" (2011):

```
Approach: Conservative rasterisation from 3 axis-aligned views

1. Setup:
   - 3D voxel texture (e.g., 256x256x256, RGBA8 for color + normal + emissive)
   - Rasterise scene mesh triangles into voxel grid

2. Conservative Rasterisation:
   - Render scene from +X, +Y, +Z orthographic views
   - Use conservative rasterisation (or geometry shader expansion)
   - For each fragment:
     - Compute voxel coordinate from fragment position
   - Store: albedo color (RGB), normal (octahedral encoding), emissive (A)

3. Implementation Options:
   - Compute shader: atomic operations on 3D texture (fastest, atomic contention at edges)
   - Rasterisation: standard render passes with orthographic cameras (simpler, 3 passes)
   - Hybrid: compute for dense meshes, rasterise for sparse

4. Memory: 256^3 * 4 bytes = 64 MB for RGBA8
```

### Voxel Mip Chain (T-GIR-P7.2)

```
1. Generate mip levels 1..8 (256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
2. Each mip level averages 2x2x2 block from the level above
3. Store in 3D texture array or individual 3D textures
4. Mip levels enable LOD cone tracing (wider cone = higher mip)
```

### Voxel Cone Tracing (T-GIR-P7.3)

```
For each pixel:
  1. Compute reflection cone from surface normal, view direction, roughness
  2. Trace cone through voxel grid:
     - Start at surface position + normal * bias
     - Step along cone axis
     - At each step, compute cone footprint -> select mip level
     - Sample voxel mip chain at current position + mip level
     - Accumulate: color *= (1 - opacity), opacity += sample_alpha
  3. Multiple cones for diffuse (wide cone, few steps) and specular (narrow cone, more steps)
  4. Cone parameters:
     - Diffuse: 3 cones (45 degree aperture), ~16 steps each, use high mip levels
     - Specular: 1 cone (aperture = roughness), ~32 steps, use low mip levels
```

## Dependencies

- T-GIR-P7.1: S16 (mesh data access)
- T-GIR-P7.2: T-GIR-P7.1
- T-GIR-P7.3: T-GIR-P7.2

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/voxelize.comp.wgsl` | Conservative voxelisation compute shader |
| `crates/renderer-backend/shaders/voxel_downsample.comp.wgsl` | Voxel mip chain generation |
| `crates/renderer-backend/shaders/voxel_cone_trace.comp.wgsl` | Voxel cone tracing compute shader |
| `crates/renderer-backend/src/voxel_gi.rs` | Voxel GI pass management |
| `engine/rendering/lighting/gi_voxel.py` | Python reference if desired |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| Scene voxelises to 256^3 grid at <3ms | Failing -- not built |
| Voxel mip chain generates correctly | Failing -- not built |
| Cone tracing produces plausible indirect light | Failing -- not built |
| Voxel GI temporal stability is acceptable | Failing -- not built |
| Voxel GI memory usage <100 MB | Failing -- not built |
