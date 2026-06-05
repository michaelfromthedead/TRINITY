# PHASE 4 ARCHITECTURE: Occlusion Culling & Meshlet Pipeline

> **Phase**: 4/7 | **Status**: [~] 10% (1 partial, 4 absent)
> **Tasks**: T-GPU-4.1 through T-GPU-4.5 (5 tasks)
> **Gaps**: S2-G1 (HZB, occlusion, meshlet culling), S2-G7 (meshlet generation)

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `engine/rendering/gpu_driven/culling.py` | 1109 | HZB constants defined (DEFAULT_HZB_WIDTH=512, etc.) |
| `engine/rendering/gpu_driven/meshlet.py` | 731 | CPU-side meshlet generation and culling |

## NOT Implemented

| File | Status |
|------|--------|
| `shaders/gpu_driven/hzb_build.comp.wgsl` | [-] Does not exist |
| `crates/.../gpu_driven/hzb.rs` | [-] Does not exist |
| `shaders/gpu_driven/gpu_cull_occlusion.comp.wgsl` | [-] Does not exist |
| `crates/.../gpu_driven/meshlet_culling.rs` | [-] Does not exist |
| `shaders/gpu_driven/gpu_cull_meshlet.comp.wgsl` | [-] Does not exist |
| `shaders/gpu_driven/gpu_cull_triangle.comp.wgsl` | [-] Does not exist |

## Reality by Task

### T-GPU-4.1: HZB construction [ - ] NOT IMPLEMENTED
No HZB mip-chain build shader exists. Constants defined in culling.py (HZB_WIDTH=512, HZB_HEIGHT=512, HZB_MIP_LEVELS=9) but no compute shader or Rust implementation.

### T-GPU-4.2: HZB occlusion culling [ - ] NOT IMPLEMENTED
No occlusion culling shader exists. No project/mip-select/sample/compare logic on GPU.

### T-GPU-4.3: Meshlet generation [ ~ ] CPU ONLY
- `engine/rendering/gpu_driven/meshlet.py` (731 lines) has:
  - Meshlet data structures (vertices, indices, bounding sphere, normal cone)
  - MeshletBuilder for partitioning meshes into meshlets
  - MeshletCuller for frustum + cone culling
- No Rust-side GPU meshlet generation
- Python implementation is for content pipeline / offline use, not runtime GPU generation

### T-GPU-4.4: Meshlet culling compute shader [ - ] NOT IMPLEMENTED
No meshlet culling compute shader exists.

### T-GPU-4.5: Triangle culling compute shader [ - ] NOT IMPLEMENTED
No triangle culling shader exists.

## Existing CPU Meshlet System (meshlet.py)

```
Meshlet (64 vertices, ~124 triangles typical):
  - bounding_sphere: center + radius for frustum cull
  - normal_cone: cone_axis + cone_angle for backface cull
  - index_offset/count into mesh index buffer

MeshletBuilder.build(mesh):
  for each triangle:
    assign to current meshlet
    if meshlet full (64 verts or ~124 tris):
      compute bounding sphere
      compute normal cone
      emit meshlet
      start new meshlet

MeshletCuller.cull(meshlets, frustum, view_dir):
  for each meshlet:
    frustum test (bounding sphere)
    backface test (normal cone vs view_dir)
    mark visible/hidden
```

## Recommended Implementation Architecture

```
GPU HZB + Occlusion + Meshlet Pipeline:

  T-GPU-4.1: HZB Build
    Input: depth buffer (full res)
    4x max reduction mip chain:
      level 0 = full res depth
      level 1 = max(4 pixels) → half res
      ...
      level N = 1x1 or min dimension
    Output: HZB texture (mip chain)
    Target: <0.3ms at 4K

  T-GPU-4.2: HZB Occlusion Cull
    Input: instance bounding spheres, HZB texture
    For each instance:
      1. Project bounding sphere to screen space
      2. Select HZB mip level (projected size heuristic)
      3. Sample HZB at projected location
      4. Compare depth: if instance is behind HZB, cull
    Conservative variant: test all 8 AABB corners

  T-GPU-4.3: Meshlet Generation (Rust)
    Offline or load-time:
      Read mesh vertex/index data
      Partition into 64-vert meshlets
      Compute bounding sphere per meshlet
      Compute normal cone per meshlet
    Output: meshlet buffer (GPU storage)

  T-GPU-4.4: Meshlet Cull
    Workgroup-per-meshlet:
      Frustum test on bounding sphere
      Backface test on normal cone vs view direction
      Optional HZB test at meshlet granularity

  T-GPU-4.5: Triangle Cull (Stretch)
    Only at LOD 0:
      Backface test per triangle
      Zero-area triangle detection
      Sub-pixel triangle culling
```
