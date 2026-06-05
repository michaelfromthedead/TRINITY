# PHASE 6 ARCH: Frame Graph Integration and Hybrid Rendering

## Status: NOT IMPLEMENTED

Phase 6 has zero implementation in the main source tree. No files exist for any of the 8 tasks.

## Design Intent (from PHASE_N_TODO.md)

Phase 6 integrates the SDF ray marching pass (S13) into the Trinity frame graph for hybrid rasterization + ray marching rendering:

### Frame Graph Registration (T-DEMO-6.1)
- Declare S13 as a single compute pass in the S1 frame graph
- Register as a full-screen pass with no vertex/geometry input

### Full-Screen Mode (T-DEMO-6.2)
- S13 writes every pixel (pure demoscene mode)
- No rasterization passes

### Hybrid Mode (T-DEMO-6.3-6.5)
- S13 reads rasterization depth buffer
- S13 writes only where closer than rasterized geometry
- Depth reconstruction from SDF hit distance

### Resource Management (T-DEMO-6.6-6.8)
- 1-2 barriers per frame between S13 and rasterization
- Multiple S13 passes (opaque + transparent)
- S13 output feeds into S8 post-processing (tone mapping, bloom, TAA)

## Prerequisites
Phase 3 (ray marching pipeline), S1 (frame graph), S8 (post-processing).
