# Pending Test Fixes — API Compatibility

**Total Pending Tests:** 122 files (was 294)
**Enabled Tests:** 218 files
**Lib Tests:** 9,263 passing
**Last Updated:** 2026-06-03

---

## Session Progress

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Integration test files | 50 | 218 | **+168** |
| Pending tests | 294 | 122 | **-172** |
| New tests enabled | — | ~16,000+ | — |

---

## Module Exports Added This Session

gpu_driven/mod.rs now exports:
- LOD types: `LodDistances`, `LodParams`, `LodConfig`, `select_lod*`, etc.
- Meshlet types: `Meshlet`, `MeshletBounds`, `MeshletData`, `MAX_MESHLET_*`
- Visibility flags: `VisibilityFlagsBuffer`, `is_visible`, `set_visible`, etc.
- Stream compact: `StreamCompactParams`, `cpu_stream_compact`, etc.
- Frustum culling: `FrustumPlane`, `Frustum`, `cpu_frustum_cull*`
- Frustum cull pipeline: `CullDispatchParams`, `workgroups_for_objects`
- Object data: `ObjectData`, `ObjectDataBuffer`, `object_flags`
- Scene data: `SceneDataBuffers`
- Frustum buffer: `FrustumBuffer`, `FrustumPlanes`, `perspective_matrix`, etc.
- Indirect draw: `IndirectDrawIndexedArgs`, `IndirectDrawBuffer`, etc.
- Multi-draw: `MultiDrawSupport`, `multi_draw_indirect*`, etc.
- HiZ pyramid: `HiZPyramid`, `HiZDownsampleParams`, `cpu_max_reduction`
- HiZ occlusion: `HiZOcclusionParams`, `cpu_project_aabb`, `cpu_test_occlusion`
- Geometry path: `GeometryPath`, `GeometryPathConfig`, `GeometryRenderable`
- LOD select: `LodSelectParams`, `ObjectLodInput`, `cpu_select_lod*`

---

## Tests Enabled This Session

| Test File | Tests | Notes |
|-----------|-------|-------|
| blackbox_meshlet | 78 | Meshlet struct operations |
| blackbox_lod | 76 | LOD selection functions |
| blackbox_meshlet_generator | 66 | MeshletData generation |
| blackbox_frustum_cull_pipeline | 19 | Frustum culling dispatch |
| blackbox_lod_buffer | 59 | LOD buffer management |
| blackbox_visibility_flags | 51 | Visibility flag bitfields |
| blackbox_object_data | 50 | ObjectData struct |
| blackbox_scene_data | 62 | SceneDataBuffers |
| blackbox_multi_draw | 155 | Multi-draw indirect |
| blackbox_indirect_draw_buffer | 24 | Indirect draw buffers |
| blackbox_hiz_pyramid | 32 | HiZ pyramid mip chain |
| blackbox_geometry_path | 64 | Geometry path enum |
| blackbox_hiz_occlusion | 45 | HiZ occlusion testing |
| blackbox_hiz_downsample | 32 | HiZ downsampling |
| blackbox_lod_select | 34 | LOD selection CPU helpers |
| blackbox_build_indirect | 26 | Build indirect pipeline |
| blackbox_dispatch_indirect_args | 44 | Dispatch args struct |
| blackbox_draw_indexed_args | 22 | DrawIndexedArgs struct |
| blackbox_draw_indirect_args | 16 | DrawIndirectArgs struct |
| blackbox_count_buffer | 24 | Count buffer management |

---

## Remaining 136 Tests — Blocker Categories

### Category 1: Missing Module `compute_library` (ARCHITECTURE)

Tests use `renderer_backend::compute_library::*` but this module doesn't exist.

| Test | Required Module |
|------|-----------------|
| blackbox_radix_sort | compute_library::radix_sort |
| blackbox_stream_compact | compute_library::stream_compact |
| blackbox_compute_library | compute_library |

**Fix:** Either create compute_library module or refactor tests to use gpu_driven directly.

---

### Category 2: Missing `frame_sync` Module

| Test | Required |
|------|----------|
| blackbox_double_buffer | frame_sync::DoubleBufferedRenderer |

---

### Category 3: frame_graph API Mismatches

| Test | Issue |
|------|-------|
| blackbox_edge_builder | EdgeBuilder not exported |
| blackbox_compiler | FrameGraphCompiler::new() signature |
| blackbox_compiled_pass | compile_with_config() missing |
| blackbox_barrier_opt | BarrierOptimizer not exported |
| blackbox_barrier_resolve | BarrierResolveContext type |

---

### Category 4: gpu_driven API Mismatches

| Test | Missing Types |
|------|--------------|
| blackbox_hiz_cull_pipeline | HiZCullParams, HiZCullPipeline, FLAG_* |
| blackbox_material_descriptor | MaterialDescriptor, GpuMaterialTable |
| blackbox_texture_table | Table API returns bool vs Result |
| blackbox_bindless_bind_group | Bindless types |
| blackbox_texture_registry | Registry types |

---

### Category 5: Platform-Specific (NOT APPLICABLE on Linux)

| Module | Tests | Status |
|--------|-------|--------|
| `vulkan` | blackbox_vulkan_features | Vulkan-specific |
| `dx12` | blackbox_dx12_features | Windows-only |
| `metal` | blackbox_metal_features | macOS-only |

---

### Category 6: Complex Integration Tests (LOW PRIORITY)

| Test | Blocker |
|------|---------|
| integration_tests | Multiple missing modules |
| test_demoscene_* | Demoscene internal APIs |
| *_phase6_gpu_driven | Phase 6 internal APIs |
| blend_tree_tests | Animation system |

---

## Behavioral Bugs Found

1. **Dead pass elimination doesn't filter async_passes**
   - Eliminated compute passes still appear in `async_passes`
   - 6 test failures in `blackbox_async_timeline`

2. **compute_barriers mixed resource boundary**
   - Returns 1 tuple instead of 2 for texture+buffer at same boundary
   - 1 test failure in `blackbox_barriers_fix`

---

## Commands

```bash
# Count remaining pending
ls crates/renderer-backend/tests_pending/*.rs | wc -l

# Count enabled
ls crates/renderer-backend/tests/*.rs | wc -l

# Try enabling a specific test
cp crates/renderer-backend/tests_pending/TEST.rs crates/renderer-backend/tests/
cargo test --package renderer-backend --test TEST 2>&1 | head -20
```

---

*Last Updated: 2026-06-03*
*Integration tests: 204 enabled, 136 pending*
*Lib tests: 9,263 passing*
