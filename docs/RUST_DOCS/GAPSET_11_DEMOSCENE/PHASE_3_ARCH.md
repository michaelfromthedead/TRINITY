# PHASE 3 ARCH: Ray Marching Compute Pipeline

## Status: NOT IMPLEMENTED

Phase 3 has zero implementation in the main source tree. No files exist for any of the 13 tasks.

## Design Intent (from PHASE_N_TODO.md)

The ray marching pipeline would be a full-screen WGSL compute shader (`@compute @workgroup_size(8, 8, 1)`) that:

1. **Camera ray generation** (T-DEMO-3.1): Generate rays from pinhole camera model using camera position, target, FOV, aspect ratio.

2. **Sphere tracing loop** (T-DEMO-3.2): March rays using the Phase 1 SDF library with:
   - Epsilon-scaled termination (T-DEMO-3.3)
   - Maximum iteration/distance bounds

3. **Normal estimation** (T-DEMO-3.4): 6-point central differences stencil.

4. **Lighting** (T-DEMO-3.5-3.8):
   - Ambient occlusion (Quilez 5-evaluation method)
   - Soft shadows (Quilez penumbra, 32 steps)
   - Lambertian diffuse (multiple lights)
   - Specular (Blinn-Phong or GGX from S3 BRDF library)

5. **Post-processing** (T-DEMO-3.10-3.13):
   - Sky color function for miss rays
   - Tone mapping (HDR to display range)
   - Depth of field (lens jitter)
   - Temporal anti-aliasing (sub-pixel jitter accumulation)

## Dependencies

- Phase 1 SDF primitives and combinators (must exist as compilable WGSL)
- S14 (RHI for compute shader dispatch)
- S3 (BRDF library for specular lighting)

## What Must Be Created

A complete WGSL compute shader (estimated 300-500 lines) that:
1. Includes all Phase 1 WGSL files via `#import` or concatenation
2. Implements the camera/ray marching/lighting pipeline
3. Outputs a full-screen rgba16f texture

No file exists at any expected path such as:
- `crates/renderer-backend/src/demoscene/ray_march.wgsl`
- `engine/rendering/demoscene/wgsl/ray_march.wgsl`
- `crates/renderer-backend/src/demoscene/demoscene_pipeline.rs`
