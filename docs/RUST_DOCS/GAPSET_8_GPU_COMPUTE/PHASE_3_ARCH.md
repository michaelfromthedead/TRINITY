# PHASE 3 ARCHITECTURE: Core Culling Pipeline

> **Phase**: 3/7 | **Status**: [~] 12% (1 partial, 3 absent)
> **Tasks**: T-GPU-3.1 through T-GPU-3.4 (4 tasks)
> **Gaps**: S2-G1 (frustum, distance/LOD culling), S2-G4 (draw-arg generation)

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `engine/rendering/gpu_driven/culling.py` | 1109 | CPU-side culling pipeline (frustum + distance + HZB config) |
| `engine/rendering/gpu_driven/indirect_draw.py` | 661 | CPU-side indirect draw structures |

## NOT Implemented

| File | Status |
|------|--------|
| `shaders/gpu_driven/gpu_cull_frustum.comp.wgsl` | [-] Does not exist |
| `engine/rendering/gpu_driven/culling.py` (GPU dispatch) | [-] CPU-only |
| `shaders/gpu_driven/gpu_cull_distance.comp.wgsl` | [-] Does not exist |
| `shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl` | [-] Does not exist |
| `crates/.../gpu_driven/indirect_draw.rs` | [-] Does not exist (Rust side) |
| `shaders/gpu_driven/` | [-] Directory does not exist |

## Reality by Task

### T-GPU-3.1: Frustum culling [ ~ ] CPU ONLY
- `culling.py` has CPU-side frustum culling with:
  - `FrustumPlanes` (6 planes: left, right, top, bottom, near, far)
  - Sphere test (`cull_sphere`), AABB test
  - `CullingPipeline` class orchestrating frustum + occlusion + distance
- No GPU compute shader (`gpu_cull_frustum.comp.wgsl`) exists
- The Python implementation would be too slow for per-frame GPU-driven culling of thousands of instances

### T-GPU-3.2: Distance/LOD culling [ - ] NOT IMPLEMENTED
- Constants defined in `culling.py`: `DEFAULT_MAX_RENDER_DISTANCE = 1000.0`, `DEFAULT_FADE_DISTANCE = 50.0`
- No GPU compute shader exists
- @lod decorator exists in `lod_streaming.py` (decorator only, no GPU feedback)

### T-GPU-3.3: Draw-arg generation [ - ] NOT IMPLEMENTED
- No GPU compute shader for batch detection + IndirectDrawIndexedArgs generation
- Python indirect_draw.py has CPU-side `MultiDrawBatch` for batch detection but no GPU generation

### T-GPU-3.4: Multi-draw indirect execution [ - ] NOT IMPLEMENTED
- No GPU indirect multi-draw with Tier 1/2/3 fallback
- No `fallback.rs`
- Python indirect_draw.py has structures but no execution engine

## Existing CPU Culling Pipeline (culling.py)

```
CullingPipeline.process_frame():
  1. Frustum cull: test each instance against 6 planes
     - Sphere test first (fast reject)
     - AABB test for survivors
  2. Distance cull: reject instances beyond MAX_RENDER_DISTANCE
     - Apply fade within FADE_DISTANCE range
  3. HZB occlusion cull: (constants only, no implementation)
     - HZB_WIDTH/HZB_HEIGHT/MIP_LEVELS defined
  4. LOD selection: based on distance bands
  5. Output: visible instance indices for indirect draw
```

This CPU pipeline would NOT satisfy GPU-driven rendering requirements (too slow for thousands-to-millions of instances per frame). A GPU compute implementation is required for all tasks.

## Recommended Implementation Architecture

```
GPU Culling Pipeline:
  [Input] Instance buffer (all instances)
     │
     ▼ T-GPU-3.1
  Frustum Cull (1 thread/instance)
     │ sphere then AABB against 6 frustum planes
     │ output: alive flags buffer
     ▼ T-GPU-3.2
  Distance Cull + LOD Select
     │ reject instances > max_draw_distance
     │ select LOD level per distance band
     │ output: LOD-selected visibility buffer
     ▼ T-GPU-3.3
  Draw-Arg Generation
     │ batch detection: group consecutive visible instances by material/mesh
     │ write IndirectDrawIndexedArgs per batch
     │ output: indirect draw command buffer
     ▼ T-GPU-3.4
  Multi-Draw Indirect (Tier 1/2/3)
     │ Tier 1: vkCmdDrawIndexedIndirectCount (best)
     │ Tier 2: CPU readback of count
     │ Tier 3: CPU batch processing
     ▼
  [Output] GPU draw calls
```
