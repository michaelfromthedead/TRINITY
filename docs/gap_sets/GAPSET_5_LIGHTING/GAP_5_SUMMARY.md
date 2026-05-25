# GAPSET_5_LIGHTING -- Independent Verification Report

**Date:** 2026-05-22
**Investigator:** Claude (deepseek-v4-flash)
**Scope:** All 33 listed tasks across 6 phases in `PHASE_N_TODO.md` (document claims 49, only 33 specified)
**Method:** Source-code inspection -- each file read, each function verified
**Corrected TODO:** `PHASE_N_TODO.md` annotated in-place below

---

## Executive Summary

The 33 checkmarks in `PHASE_N_TODO.md` are all untoggled `[ ]`. After deep source-code verification: **1 item is REAL** (exists as described), **4 are PARTIAL** (exist but different/incomplete), **28 are ABSENT** (do not exist).

The project has substantial lighting infrastructure but it took a **Python-native path** with WGSL shaders that are partially built. The Python modules (`engine/rendering/lighting/`) implement the full reference design with all 7 light types, CSM/cube/spot shadow maps, 5 shadow filtering techniques, DDGI, and light probes. The WGSL shaders implement a **forward PBR** path (not the deferred froxel-culled pipeline the TODO describes), with 3 light types, basic CSM with PCF, and a DDGI compute shader.

**Critical architectural divergence:** The TODO describes a deferred compute-shader-based renderer with froxel light culling. The actual WGSL implements forward fragment-shader-based PBR with direct light iteration. The `light_culling.wgsl` shader exists but is not consumed -- `pbr.frag.wgsl` iterates all lights directly.

---

## Per-Phase Verdict Summary

| Phase | Tasks | [x] | [~] | [-] | Verdict |
|-------|-------|-----|-----|-----|---------|
| 1: GPU Light Data Infrastructure | 5 | 0 | 0 | 5 | **ABSENT** -- no `light_types.rs`, no SoA builder |
| 2: Froxel Clustered Culling | 5 | 0 | 2 | 3 | **PARTIAL** -- WGSL shader exists but non-functional (unconsumed) |
| 3: Deferred PBR Lighting | 7 | 1 | 0 | 6 | **PARTIAL** -- forward PBR fragment shader exists, deferred compute absent |
| 4: Cascaded Shadow Maps | 6 | 0 | 2 | 4 | **PARTIAL** -- WGSL CSM sampling works, Rust dispatch absent |
| 5: Cube + Spot Shadow Maps | 5 | 0 | 0 | 5 | **ABSENT** -- no WGSL for cube/spot shadows |
| 6: Shadow Atlas + Filtering | 5 | 0 | 0 | 5 | **ABSENT** -- no WGSL for atlas or filter modules |

---

## Detailed Task-by-Task Findings

### Phase 1: GPU Light Data Infrastructure (T-LIT-1.1 through 1.5)

**T-LIT-1.1** `light_types.rs` with LightTypeGPU, LightUnion, 7 repr(C) types:
[-] No `light_types.rs` exists anywhere in `crates/renderer-backend/src/`. `pbr.frag.wgsl` has inline `DirectionalLight`, `PointLight`, `SpotLight` structs (3 types, not 7). No `LightTypeGPU` enum. No `LightUnion` tagged union. Python `light_types.py` defines all 7 types as dataclasses with decorators.

**T-LIT-1.2** SoA buffer builder:
[-] No SoA buffer builder exists. Lights in WGSL are AoS arrays (`array<PointLight>`).

**T-LIT-1.3** `lighting_system.py` orchestrator:
[-] No `lighting_system.py` exists in `engine/rendering/lighting/`. The 7 Python files are independent modules with no orchestrator.

**T-LIT-1.4** CPU-to-GPU upload path with staging ring:
[-] No lighting-specific upload path. `gpu_driven/buffers.rs` (24,809 bytes) has generic GPU buffer staging but nothing light-specific.

**T-LIT-1.5** Bind group layout for light data buffers:
[-] No standalone bind group builder. `pbr.frag.wgsl` has inline bindings at group(2) for lights and group(3) for shadows, but no dedicated module.

### Phase 2: Froxel Clustered Culling (T-LIT-2.1 through 2.5)

**T-LIT-2.1** Froxel AABB reconstruction in WGSL:
[~] `light_culling.wgsl` (229 lines) EXISTS. Computes simplified froxel AABBs using view-space far_depth estimates rather than proper inverse-projection corner reconstruction. The AABB formula `vec3<f32>(-far_depth, -far_depth, near_depth)` to `vec3<f32>(far_depth, far_depth, far_depth)` is a rough bounding box, not accurate froxel corners. Uses exponential depth slicing. Python `FroxelGrid._compute_froxel_bounds()` does proper frustum corner unprojection.

**T-LIT-2.2** Per-froxel light intersection (6 types):
[~] Implements sphere-AABB for point lights and cone-AABB for spot lights (2 of 6 types). No handling for directional (should be in all froxels), IES, area, or sky lights. Python `ClusteredLightCuller._cull_light()` handles all 7 types including DirectionalLight (all froxels) and area lights.

**T-LIT-2.3** Atomic index buffer compaction:
[-] No atomic counter used. Uses pre-computed offsets from `froxel_grid[].light_offset`. No overflow handling. `MAX_LIGHTS_PER_FROXEL = 64` (spec asks for 128). No mechanism to drop farthest lights on overflow.

**T-LIT-2.4** Rust dispatch in `culling.rs`:
[-] No `culling.rs` file exists in the crate.

**T-LIT-2.5** GPU-readback correctness test:
[-] No GPU-readback test exists for light culling.

### Phase 3: Deferred PBR Lighting (T-LIT-3.1 through 3.7)

**T-LIT-3.1** `light_eval.wgsl` shared module:
[-] No `light_eval.wgsl` file exists. BRDF functions are inlined in `pbr.frag.wgsl` (lines 118-316).

**T-LIT-3.2** Directional + point + spot light evaluation in WGSL:
[x] `pbr.frag.wgsl` contains `eval_directional_light()`, `eval_point_light()`, `eval_spot_light()` functions. Each computes distance attenuation (smooth falloff `pow(clamp(1-d^2/r^2), 2)`), cone attenuation for spots (smoothstep), and delegates to a shared `eval_brdf()` with Cook-Torrance BRDF (GGX NDF, Smith-GGX geometry, Schlick Fresnel). Directional shadows via `shadow_factor()`. Structurally complete for these 3 types.

**T-LIT-3.3** LTC LUT generation:
[-] No `ltc_lut.py` or LTC compute shader exists.

**T-LIT-3.4** Area light evaluation via LTC in WGSL:
[-] No RectAreaLight or DiskAreaLight evaluation in any WGSL file. Python `RectAreaLight` and `DiskAreaLight` exist but only in Python reference.

**T-LIT-3.5** IES light evaluation via 2D texture:
[-] No IES light handling in any WGSL file. Python `IESLight` with `IESProfile` exists.

**T-LIT-3.6** `lighting_pass.comp.wgsl`:
[-] No compute shader exists. `pbr.frag.wgsl` is a forward fragment shader, not a deferred compute pass. No G-buffer reading, no froxel resolution, no HDR accumulation compute.

**T-LIT-3.7** Rust dispatch in `lighting_pass.rs`:
[-] No `lighting_pass.rs` exists.

### Phase 4: Cascaded Shadow Maps (T-LIT-4.1 through 4.6)

**T-LIT-4.1** PSSM cascade split computation in `csm.rs`:
[-] No `csm.rs`. Python `CascadedShadowMap._compute_cascade_splits()` EXISTS with logarithmic/linear blend (lambda=0.75). Default distances: [10, 30, 100, 500].

**T-LIT-4.2** Frustum fitting + texel snapping in `csm.rs`:
[-] No `csm.rs`. Python `_compute_cascade_matrices()` fits frustum corners to orthographic projection. `_stabilize_bounds()` snaps to texel boundaries to prevent shimmer.

**T-LIT-4.3** `shadow_csm.vert.wgsl` and `shadow_csm.frag.wgsl`:
[~] Files exist as `shadow.vert.wgsl` (35 lines) and `shadow.frag.wgsl` (12 lines) -- different naming. `shadow.vert.wgsl` has `CascadeUniforms` with `light_view_proj` matrix and `ModelUniforms`. Transforms vertices to light clip space. `shadow.frag.wgsl` is a no-op (depth written automatically). Functionally equivalent to CSM depth rendering.

**T-LIT-4.4** Cascade selection + blend in lighting pass:
[~] `shadow_csm.wgsl` (161 lines) implements `select_cascade()` with view-space depth threshold iteration. `pbr.frag.wgsl` has `shadow_factor()` integrating cascade selection. **No cascade blend range** -- transitions are hard switches. Python specifies `cascade_blend_range=2.0` but WGSL has no blending.

**T-LIT-4.5** CSM atlas layout (2x2 grid):
[-] WGSL uses `texture_depth_2d_array` with 4 independent layers, not an atlas. Python `ShadowAtlas` (2D bin-packing) EXISTS for general shadow map packing but is not used for CSM specifically.

**T-LIT-4.6** Multi-viewport cascade rendering in Rust:
[-] No cascade rendering dispatch in Rust. No multi-viewport or sequential draw calls.

### Phase 5: Cube + Spot Shadow Maps (T-LIT-5.1 through 5.5)

**T-LIT-5.1** `shadow_cube.vert.wgsl` and `shadow_cube.frag.wgsl`:
[-] No cube shadow WGSL files. Python `CubeShadowMap` EXISTS with 6 face view matrices using `CUBE_FACE_DIRECTIONS` list and `Mat4.look_at`.

**T-LIT-5.2** 2D array rendering for cube faces:
[-] No array texture or layer-based cube rendering in WGSL. Python has face matrices but no GPU-side cube map implementation.

**T-LIT-5.3** `shadow_spot.vert.wgsl` and `shadow_spot.frag.wgsl`:
[-] No spot shadow WGSL files. Python `SpotShadowMap` EXISTS with `Mat4.perspective` matching cone FOV.

**T-LIT-5.4** Cone penumbra fade at spot boundary:
[-] No cone fade in WGSL. Python `SpotLight.get_angular_attenuation()` has smoothstep between cos_inner and cos_outer.

**T-LIT-5.5** `cube.rs` + `spot.rs` dispatches:
[-] No cube or spot dispatch Rust files.

### Phase 6: Shadow Atlas + Filtering (T-LIT-6.1 through 6.5)

**T-LIT-6.1** `atlas.rs` bin-packing allocator:
[-] No `atlas.rs` in Rust. Python `ShadowAtlas` EXISTS with best-fit rectangle packing, allocate/deallocate/defragment. Four tile sizes available. 4096 default resolution.

**T-LIT-6.2** ShadowTileInfo GPU buffer:
[-] No ShadowTileInfo GPU structure in any WGSL. Python `ShadowAtlasSlot` with `uv_offset`/`uv_scale` properties exists but has no GPU counterpart.

**T-LIT-6.3** `shadow_common.wgsl`:
[-] No `shadow_common.wgsl` exists. Bias computation is inlined in `shadow_csm.wgsl` and `pbr.frag.wgsl`.

**T-LIT-6.4** `shadow_filter_pcf.wgsl`:
[-] No separate PCF module. PCF is inlined in `shadow_csm.wgsl::pcf_sample()` (lines 91-121) and `pbr.frag.wgsl::shadow_factor()` (lines 160-206). Python `PCFFilter` supports grid/Poisson/Vogel patterns with 4 kernel sizes.

**T-LIT-6.5** `shadow_filter_pcss.wgsl`:
[-] No PCSS WGSL module. Python `PCSSFilter` EXISTS with 3-step blocker search, penumbra estimation, and variable PCF.

---

## Cross-Reference: GAPSET_3_BRIDGE Shared Infrastructure

GAPSET_3_BRIDGE built the shared infrastructure that GAPSET_5_LIGHTING depends on:

| GAP 3 Artifact | Status | Used by GAP 5? |
|----------------|--------|----------------|
| `light_culling.wgsl` | REAL (229 lines) | Exists but unconsumed -- no pass reads froxel output |
| `shadow_csm.wgsl` | REAL (161 lines) | Used by `pbr.frag.wgsl` shadow_factor() |
| `ddgi.wgsl` | REAL (240 lines) | Has Rust pass builder in `ddgi.rs` but no runtime execution |
| `pbr.frag.wgsl` | REAL (377 lines) | Main lighting shader -- forward PBR, not deferred |
| `gpu_driven/buffers.rs` | REAL | Generic GPU staging -- no light-specific usage |
| `frame_graph/` | REAL | Pass IR exists but no lighting passes wired |
| `renderer.rs` | PARTIAL | Only triangle rendering -- no lighting pipeline |

The bridge pattern: GAP 3 built the WGSL shaders and Rust infrastructure. GAP 5 was supposed to build the light data pipeline, compute dispatches, and integrate everything. The WGSL shaders from GAP 3 are real but the integration scaffolding from GAP 5 does not exist.

---

## Verification Notes

- All WGSL shaders verified by reading each file in its entirety
- Python reference modules verified by reading each `.py` file
- Rust modules verified by searching for file existence and reading key files
- No test harness was run -- verification is static code analysis only
- The claim of 49 total tasks is inaccurate; only 33 tasks are specified in the document
