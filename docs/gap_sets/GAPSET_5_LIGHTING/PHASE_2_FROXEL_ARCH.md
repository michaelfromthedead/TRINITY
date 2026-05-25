# Phase 2: Froxel Clustered Culling Architecture

## Current Status: PARTIAL (2/5 tasks partial, 3 absent)

`light_culling.wgsl` (229 lines) exists but uses simplified AABBs and is not consumed by any shader. No Rust dispatch.

## Architecture

### Froxel Grid Configuration

| Parameter | Python Default | WGSL Current | TODO Spec |
|-----------|---------------|--------------|-----------|
| Tiles X | 16 | 16 | 16 |
| Tiles Y | 9 | 16 | 9 |
| Depth Slices | 24 | 32 | 24 |
| Near plane | 0.1 | 0.1 | 0.1 |
| Far plane | 1000 | - | 1000 |
| Max lights/froxel | -- | 64 | 128 |
| Tile size | auto | 16 | 16 |

### Fixes Required for light_culling.wgsl

1. **Proper froxel AABB reconstruction**: Replace simplified `vec3(-far_depth, ...)` with full unprojection from NDC to view space using the inverse projection matrix (as in Python `FroxelGrid._compute_froxel_bounds()`).

2. **Add light type dispatch**: Use a `LightType` buffer or counter to branch on all 7 types:
   - DirectionalLight -> add to ALL froxels unconditionally
   - PointLight -> sphere-AABB test (already exists)
   - SpotLight -> cone-AABB test (already exists, needs refinement)
   - RectAreaLight -> sphere approximation using diagonal radius
   - DiskAreaLight -> sphere approximation using disk radius
   - IESLight -> sphere-AABB test (treat as oriented point light)
   - SkyLight -> add to all froxels

3. **Atomic index compaction**: Replace pre-computed offset with `atomicAdd` on a global counter. Handle overflow by dropping farthest lights.

4. **Workgroup strategy**: Current single-thread-per-workgroup design is a bottleneck. Use workgroup-reduce pattern with shared memory for parallel light testing.

### Rust Dispatch (culling.rs)

```
culling.rs:
  struct FroxelCullingPass {
      dispatch: (tiles_x, tiles_y, 1),
      buffers: [light_buffer, froxel_grid, index_list, counter],
      params: CullingParams uniform,
  }
```

Must support 3 grid configurations:
- 16x9x24 (default, matches Python)
- 12x8x16 (lower res for mobile/VR)
- 8x4x12 (minimal for debug)

### Integration Path

1. Fix `light_culling.wgsl` AABBs and intersection tests
2. Build `culling.rs` dispatch
3. Wire froxel output into `pbr.frag.wgsl` via storage buffer (read light_index_list per fragment)
4. Verify against Python `ClusteredLightCuller` output

### Test Plan (Python Reference)

```
ClusteredLightCuller ref_culler = ClusteredLightCuller(FroxelGrid(config))
ref_culler.set_lights(test_scene_lights)
ref_culler.cull()
ref_lists = ref_culler.get_light_lists()

GPU culler runs equivalent compute shader
Readback froxel_grid and light_index_list
Assert per-froxel light counts and indices match
```

4 test scenes: simple (few lights), complex (all types), overflow (128+ lights per froxel), empty (no lights).
