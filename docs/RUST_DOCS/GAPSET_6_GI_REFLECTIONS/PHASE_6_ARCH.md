# Phase 6: Planar Reflections -- Architecture

## Overview

Phase 6 implements planar (mirror) reflections: rendering reflected geometry from a mirrored camera and applying oblique near-plane clipping for optimal depth precision.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P6.1 | [-] | Implement planar mirror rendering |
| T-GIR-P6.2 | [-] | Implement oblique near-plane clipping |

## Current State: NOT BUILT

No planar reflection code exists anywhere in the codebase.

## Required Architecture

### Planar Mirror Rendering (T-GIR-P6.1)

Design per UE4 planar reflections and standard mirror rendering technique:

```
1. Mirror Setup:
   - PlanarMirror component defines: mirror plane equation (normal + distance), render resolution, LOD bias
   - Mirrors can be static (baked) or dynamic (per-frame)

2. Reflected Camera:
   - Reflect camera position across mirror plane: pos' = reflect(pos, plane)
   - Reflect camera direction across mirror plane: dir' = reflect(dir, plane)
   - Compute reflected view matrix from pos' and dir'
   - Flip winding order (mirror reverses triangle orientation)

3. Render Pass:
   - Create temporary render target (typically half-resolution RGBA16F)
   - Cull geometry to mirror frustum (clip against mirror plane)
   - Render scene from reflected camera viewpoint
   - Apply fade at reflection edges (screen edge fade, distance fade)

4. Sampling:
   - In main pass, compute mirror UV from reflected camera projection
   - Sample mirror render target at computed UV
   - Apply roughness-based blur to reflection
```

### Oblique Near-Plane Clipping (T-GIR-P6.2)

Standard technique from Lengyel, "Oblique View Frustum Clipping" (2005):

```
1. Compute the mirror plane in camera space
2. Modify the projection matrix so the near plane coincides with the mirror plane
3. This clips geometry behind the mirror, preventing artifacts
4. The modified projection matrix is:
   - Compute clip-space mirror plane: C = transpose(inverse(M)) * P_mirror
   - Modify projection matrix third row: M[2] = C / C.w
   - Apply to the reflected camera's projection matrix
```

## Dependencies

- T-GIR-P6.1: S1 (frame graph infrastructure)
- T-GIR-P6.2: T-GIR-P6.1

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/src/planar_reflections.rs` | Planar mirror component and pass management |
| `crates/renderer-backend/shaders/planar_reflection_blit.wgsl` | Blit/transform from mirror RT to main buffer |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| Mirror reflects scene geometry correctly | Failing -- not built |
| Oblique clipping prevents back-mirror artifacts | Failing -- not built |
| Multiple mirrors render independently | Failing -- not built |
| Mirror reflection handles roughness correctly | Failing -- not built |
