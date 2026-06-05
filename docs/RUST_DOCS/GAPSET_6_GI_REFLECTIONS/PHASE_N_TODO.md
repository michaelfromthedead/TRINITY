# GAPSET 6: Global Illumination & Reflections -- Phase Task List

> **Cluster**: GAPSET_6_GI_REFLECTIONS
> **Task ID Format**: T-GIR-P{PHASE}.{N}
> **Total Tasks**: 45 across 11 phases
> **33 Gap Coverage**: All S6-G1 through S6-G19 and S7-G1 through S7-G14

---

## Phase 1: Foundation Infrastructure (5 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P1.1 | Implement spherical harmonics math library: Python (numpy) reference + WGSL compute functions for 3rd-order SH (9 coefficients irradiance, 4 coefficients visibility). Functions: sh_evaluate, sh_project, sh_rotate, sh_convolve_irradiance. | S6-G3, S6-G12 | Medium | None |
| T-GIR-P1.2 | Define probe GPU storage buffers: ProbeGrid (positions, cell size, origin), ProbeSH (irradiance + visibility coefficients per probe), ProbeVis (occlusion terms). Rust structs in crates/renderer-backend/src/gi/probe_grid.rs. Ring buffer for scrolling volumes. | S6-G2 | Medium | T-GIR-P1.1 |
| T-GIR-P1.3 | Extend @reflection_probe decorator with 8 missing parameters: importance, box_extents, inner/outer_radius, roughness_levels, blend_distance, capture_lod_bias, include_layers, exclude_actors. Add bounding box fields to ReflectionProbeConfig. | S7-G12 | Small | None |
| T-GIR-P1.4 | Define GI performance budget table: per-tier timings (Low 0.2ms through Cinematic 12.7ms), GPU timestamp query instrumentation, 3-frame hysteresis fallback logic, budget monitoring module in engine/rendering/gi/gi_config.py. | S6-G17 | Small | T-GIR-P1.2 |
| T-GIR-P1.5 | Define reflection buffer format: ReflectionBuffer struct (color, roughness, hit_distance, technique_mask), half-res default resolution, bilateral upscale shader specification. | S7-G11 | Small | None |

**Phase 1 acceptance**: SH library passes numerical validation (Ramamoorthi & Hanrahan 2001 reference). Probe GPU buffers compile with correct stride and alignment. Decorator extensions pass existing test suite. Budget monitor correctly triggers fallback after 3 consecutive over-budget frames.

---

## Phase 2: DDGI Core -- Uniform Grid (8 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P2.1 | Implement DDGI probe placement system: camera-relative uniform 3D grid, CPU-side grid origin computation, configurable spacing (4-8m), fixed-size GPU allocation (32x32x4 High, 128x128x16 Ultra). Python orchestration in engine/rendering/lighting/gi_ddgi.py. | S6-G1 | Medium | T-GIR-P1.2 |
| T-GIR-P2.2 | Implement DDGI probe ray tracing -- hardware RT path: ray generation per probe via S10 ray queries, N rays (32-128) stratified spherical distribution, hit point radiance accumulation. Shader: ddgi_probe_update.comp.wgsl. | S6-G7 | Large | T-GIR-P2.1, S10 TLAS |
| T-GIR-P2.3 | Implement DDGI probe ray tracing -- rasterised fallback: six 90-degree face maps per probe, G-buffer read for albedo/normal, atlas batching for draw call minimisation. | S6-G1 | Large | T-GIR-P2.1 |
| T-GIR-P2.4 | Implement DDGI probe update: irradiance accumulation (weighted by distance gaussian + confidence), visibility minimum-distance storage, temporal accumulation (lerp over 8-32 frames). Importance-based update rate (critical=every frame to low=every 16 frames). | S6-G13, S6-G15 | Medium | T-GIR-P2.2 |
| T-GIR-P2.5 | Implement DDGI probe sampling at shading point: find 8 nearest probes, trilinear weights, SH irradiance evaluation in direction of shading normal, visibility modulation, parallax correction via wall-normal weighting. Shader: ddgi_probe_sampling.wgsl. | S6-G13 | Medium | T-GIR-P1.1, T-GIR-P2.1 |
| T-GIR-P2.6 | Implement DDGI infinite scrolling volumes: fixed-size GPU ring buffer, per-frame grid origin computation, probe slot re-indexing, seed-new-probes from neighbours with noise dither. Shader: ddgi_grid_shift.comp.wgsl. | S6-G1 | Medium | T-GIR-P2.1 |
| T-GIR-P2.7 | Implement radiance cache: 64x64x32 3D grid texture, temporal accumulation from probe data, shader-based update. Shader: radiance_cache_update.comp.wgsl. | S6-G5 | Medium | T-GIR-P2.4 |
| T-GIR-P2.8 | Implement irradiance volume system: multiple independent probe grids with cross-fade blending at volume boundaries, IrradianceVolumeManager in engine/rendering/gi/irradiance_volumes.py. | S6-G19 | Medium | T-GIR-P2.1 |
| T-GIR-P2.9 | Implement light probe lightmap baker: editor tool for baking static light probes into SH coefficients via offline trace, output to .ktx2 or custom format. | S6-G18 | Large | T-GIR-P1.1 |

**Phase 2 acceptance**: 32-probe DDGI grid updates in <1.5ms. Trilinear interpolation produces linear gradients between known probe values. Scrolling grid preserves data in overlap region with no visible pop-in. Irradiance volume cross-fade is seamless.

---

## Phase 3: SSGI -- Screen-Space GI (2 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P3.1 | Implement SSGI ray marching: HiZ-accelerated depth buffer ray march, cosine-weighted hemisphere sampling (4-16 rays per pixel at half res), HZB mip chain generation, hit accumulation with distance fade (0 contribution beyond 10-20m). Shader: ssgi_trace.comp.wgsl. | S6-G14 | Medium | T-GIR-P4.1 (HiZ) |
| T-GIR-P3.2 | Implement SSGI temporal accumulation: reprojection via velocity buffer (S8), neighbourhood clamping, disocclusion reset. Shader: ssgi_temporal.comp.wgsl. | S6-G15 | Medium | T-GIR-P3.1, S8 velocity |

**Phase 3 acceptance**: SSGI traces correctly find visible hit quads. Off-screen fade reaches zero at frustum boundary and distance limit. Temporal accumulation shows reducing variance over 8+ frames. Disocclusion correctly resets history.

---

## Phase 4: SSR Core -- Screen-Space Reflections (5 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P4.1 | Implement HiZ buffer generation: mip chain of G-Buffer depth, maximum depth per texel, floor(log2(max(W,H))) levels. Shared struct with S2. Shader: hiz_generate.comp.wgsl. | S7-G2 | Medium | S1 (frame graph dispatch) |
| T-GIR-P4.2 | Implement SSR HiZ ray marching: start at coarse mip, 8-12 steps per level, descend on depth approach, binary search refinement. Shader: ssr_ray_march.comp.wgsl. | S7-G1 | Large | T-GIR-P4.1 |
| T-GIR-P4.3 | Implement SSR linear ray marching fallback: fixed 4-8 pixel stride, binary search refinement. Shader: ssr_ray_march_linear.comp.wgsl. Edge/distance/frustum fade function. Shader: ssr_fade.wgsl. | S7-G1 | Medium | T-GIR-P4.1 |
| T-GIR-P4.4 | Implement SSR temporal reprojection: velocity buffer reprojection, confidence-weighted blend, disocclusion rejection (depth delta, normal dot, velocity magnitude). Ping-pong history buffers. Shader: ssr_temporal.comp.wgsl. | S7-G9 | Medium | T-GIR-P4.2, S8 velocity |
| T-GIR-P4.5 | Implement SSR roughness-driven blur: Bloomberg-style multi-downsample, separable Gaussian with `kernel = roughness^2 * max_radius`, edge-aware bilateral upscale. Define material reflection parameters (intensity, roughness_offset, technique_override) for PBRMaterial. | S7-G13 | Medium | T-GIR-P4.4 |

**Phase 4 acceptance**: HiZ ray march completes in 40-60 steps (vs 256+ linear) with correct intersection. Temporal reprojection eliminates flickering after 8 accumulated frames. Blur kernel radius scales correctly with roughness. Edge fade reaches 0 at screen borders.

---

## Phase 5: Reflection Probe System (6 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P5.1 | Implement baked probe capture: offline cubemap rendering, 6 faces at 90 FOV, BC6H compression, .ktx2 storage, mip chain + pre-filtered roughness levels. Loading via S16 asset pipeline. | S7-G3 | Medium | S16 (asset loading) |
| T-GIR-P5.2 | Implement realtime probe capture: 6-face rendering amortised over frames, face_to_render scheduler, LOD-biased geometry, dynamic object inclusion. CPU scheduler in engine/rendering/lighting/reflection_probes.py. | S7-G3 | Large | T-GIR-P5.1 |
| T-GIR-P5.3 | Implement probe blending: per-pixel collection of influencing probes, distance*normal*visibility weighting, weight normalisation, cubemap blend. Shader: probe_blend.comp.wgsl. | S7-G4 | Medium | T-GIR-P5.2 |
| T-GIR-P5.4 | Implement parallax correction: box projection algorithm (UE4/Lagarde), reflection ray intersection with probe bounding box, corrected cubemap sampling direction. Shader: probe_parallax_correction.wgsl. | S7-G5 | Medium | T-GIR-P5.2 |
| T-GIR-P5.5 | Implement pre-filtered cubemaps: GGX distribution filter per roughness level (8-10 levels), split-sum approximation, storage as additional cubemap mips. Shader: probe_prefilter.comp.wgsl. | S7-G6 | Medium | T-GIR-P5.2 |
| T-GIR-P5.6 | Implement probe atlas: fixed-grid atlas packing (e.g., 4x4 probes), per-probe atlas UV coordinates, atlas update on capture. Management module: engine/rendering/lighting/probe_atlas.py. | S7-G14 | Medium | T-GIR-P5.2 |

**Phase 5 acceptance**: Baked probe loads and samples correctly. Realtime probe captures 6 faces over 6 frames at specified resolution. Blend weight at midpoint of two equal probes is 0.5/0.5. Box projection corrects cubemap sampling direction for offset shading point. Pre-filtered cubemap produces blurrier reflections at higher roughness.

---

## Phase 6: Planar Reflections (2 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P6.1 | Implement planar mirror rendering: reflected camera setup, mirror plane render pass, Fresnel falloff, per-mirror resolution scaling, PlanarMirror component definition. | S7-G10 | Medium | S1 (frame graph) |
| T-GIR-P6.2 | Implement oblique near-plane clipping: modify projection matrix to clip near plane to mirror plane, prevent rendering geometry in front of mirror. | S7-G10 | Medium | T-GIR-P6.1 |

**Phase 6 acceptance**: Planar mirror reflects scene correctly from reflected camera viewpoint. Oblique clipping prevents geometry in front of mirror plane from appearing in reflection. Fresnel falloff reduces reflection at normal incidence. Max 2 active planar mirrors per frame enforced.

---

## Phase 7: Voxel GI (3 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P7.1 | Implement scene voxelisation: conservative rasterisation compute shader, AABB-triangle intersection in voxel space, albedo+emissive+normal write, opacity classification. Resolution tiers: 64^3 through 256^3. Shader: voxelize.comp.wgsl. | S6-G6 | Large | S16 (mesh data) |
| T-GIR-P7.2 | Implement voxel mip chain and storage: 3D texture with mip levels (average 8 children), radiance + opacity, variance at high mips. Storage abstraction for wgpu 3D texture. Shader: voxel_downsample.comp.wgsl. | S6-G4 | Medium | T-GIR-P7.1 |
| T-GIR-P7.3 | Implement voxel cone tracing: 6-12 diffuse cones (wide aperture), 1-4 specular cones (narrow aperture), exponential step spacing, mip selection by cone aperture, opacity-weighted front-to-back compositing. Shader: voxel_cone_trace.comp.wgsl. Shared cone trace utility: voxel_cone_trace.wgsl. | S6-G4 | Large | T-GIR-P7.2 |

**Phase 7 acceptance**: Conservative voxelisation correctly fills voxels for a single test triangle. Mip chain correctly averages 8 child voxels into parent. Cone tracing accumulates occlusion through opaque wall and terminates. 256^3 voxelisation completes within 4ms.

---

## Phase 8: RT Reflections (5 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P8.1 | Implement RT reflection ray generation: read G-Buffer per pixel, reconstruct world position, compute reflection direction, trace against TLAS, roughness-based skip threshold (0.7). Shader: rt_reflections.rgen. | S7-G7 | Large | S10 (TLAS, SBT) |
| T-GIR-P8.2 | Implement BRDF importance sampling for RT reflections: GGX microfacet distribution sampling, half-vector generation, tangent basis construction, world-space transform. BRDF evaluation at hit point. Shader: rt_reflections.rchit. | S7-G7 | Medium | T-GIR-P8.1 |
| T-GIR-P8.3 | Implement roughness-based ray count adaptation: 1 ray/pixel (smooth) through quarter-res heavy denoise (rough), resolution hierarchy, adaptive scheduling. | S7-G7 | Medium | T-GIR-P8.1 |
| T-GIR-P8.4 | Implement RT reflection denoising: A-trous wavelet spatial filter (4-5 iterations, depth/normal/luminance edge-stopping), temporal accumulation (8-16 frames), bilateral upscale from half-res. Shader: rt_reflections_denoise.comp.wgsl. | S6-G8, 3.9 | Large | T-GIR-P8.1, S8 velocity |
| T-GIR-P8.5 | Implement reflection fallback chain: per-pixel RT -> SSR -> Reflection Probes -> Environment Map decision pipeline, confidence-based blending between tiers, smooth transitions (no popping). Shader: reflection_fallback_chain.comp.wgsl. | S7-G8 | Medium | T-GIR-P4.2, T-GIR-P5.3, T-GIR-P8.1 |

**Phase 8 acceptance**: RT reflections produce correct mirror reflection on smooth surfaces. BRDF importance sampling matches GGX distribution (verified via Monte Carlo integration). Roughness-based ray count correctly reduces samples for rough surfaces. Fallback chain transitions from RT->SSR on RT miss, SSR->probe on miss, probe->env on all miss. Denoised RT output converges to stable reflection after 16 frames.

---

## Phase 9: Denoising Infrastructure (3 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P9.1 | Implement A-trous wavelet spatial denoiser: 4-5 iterations with increasing dilation (1,2,4,8,16), edge-stopping functions (depth exponential, normal dot, YCoCg luminance), ping-pong buffers. Shared edge-stop library: denoise_edge_stop.wgsl. Shader: denoise_atrous.comp.wgsl. | S6-G8, 3.9 | Medium | S8 (velocity buffer) |
| T-GIR-P9.2 | Implement temporal denoiser: reprojection via velocity buffer, variance-guided accumulation, exponential moving average, neighbourhood clamping, history length tracking (1-64 frames). Shared state management in Rust: denoise_state.rs. Shader: denoise_temporal.comp.wgsl. | S6-G8, 3.9 | Medium | T-GIR-P9.1, S8 velocity |
| T-GIR-P9.3 | Implement SVGF-style variance estimation: 5x5 neighbourhood luminance mean+variance, spatiotemporal variance-guided filtering. Research task: assess SVGF vs simpler A-trous for TRINITY's use cases (path tracing reference, RT GI, RT reflections). | S6-G8, 3.9 | Large | T-GIR-P9.1, T-GIR-P9.2 |

**Phase 9 acceptance**: A-trous filter removes Monte Carlo noise while preserving edges (measured PSNR improvement vs. unfiltered). Temporal accumulator converges to stable image within 16 frames on static scene. SVGF path (if chosen) shows >2dB PSNR improvement over A-trous alone.

---

## Phase 10: GI Visualization (1 Task)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P10.1 | Implement GI debug visualization overlay: probe grid positions (colour-coded by irradiance), voxel occupancy wireframe, SSGI confidence heatmap, path tracer comparison difference heatmap, reflection technique mask per pixel. Debug-only frame graph passes. Toggle via @debug decorator. | S6-G16 | Medium | T-GIR-P2.1, T-GIR-P3.1, T-GIR-P8.5 |

**Phase 10 acceptance**: All debug views render correctly in development builds. Probe grid overlay shows correct positions with colour gradient matching irradiance. Path tracer comparison heatmap highlights >10% difference as red. Reflection technique mask correctly identifies per-pixel tier.

---

## Phase 11: Advanced GI Research (3 Tasks)

| TASK_ID | Description | Gaps | Effort | Dependencies |
|---------|-------------|------|--------|-------------|
| T-GIR-P11.1 | DDGI adaptive probe placement: research and prototype. Survey literature (Frostbite, UE5 proprietary methods). Prototype CPU-based adaptive placement: start with coarse 16^3 grid, subdivide cells where probe-to-probe irradiance variance exceeds threshold. Evaluate temporal stability. Deliverable: algorithm specification with WGSL pseudocode. | S6-G9 | Large (Research) | T-GIR-P2.1 |
| T-GIR-P11.2 | Sparse Voxel Octree implementation: research and prototype. Follow Crassin 2011 / Laine 2010 methodology. Build CPU-based SVO constructor: full 256^3 -> compress to SVO -> mip chain. Profile memory savings (target: 5-10x over dense 256^3). Evaluate GPU conversion feasibility. Deliverable: decision document with GPU implementation plan. | S6-G10 | Large (Research) | T-GIR-P7.1 |
| T-GIR-P11.3 | Lumen-Lite feasibility study: assess components (mesh cards, screen probes, radiance cache, software SDF tracing) against TRINITY's existing DDGI infrastructure. Evaluate mesh card generation pipeline cost. Determine SDF construction feasibility. Compare expected quality against DDGI at equivalent cost. Deliverable: "go/no-go" decision document with prototype plan. | S6-G11 | Medium (Research) | T-GIR-P2.4, T-GIR-P2.7 |

**Phase 11 acceptance**: Each research task produces a decision document with clear recommendation, algorithm pseudocode, estimated implementation effort, and risk assessment. Adaptive placement prototype demonstrates correct subdivision on at least 3 test scenes. SVO prototype achieves target compression ratio. Lumen-Lite study reaches clear go/no-go conclusion.

---

## Phase Dependency Graph

```
Phase 1 (Foundation)
  |
  v
Phase 2 (DDGI Core) ---> Phase 3 (SSGI) ---> Phase 10 (Visualization)
  |                         
  |                         v
  |                      Phase 4 (SSR Core)
  |                         |
  |                         v
  |                      Phase 5 (Reflection Probes)
  |                         |
  |                         v
  |                      Phase 6 (Planar Reflections)
  |
  +---> Phase 7 (Voxel GI) ---> Phase 11.2 (SVO Research)
  |
  +---> Phase 8 (RT Reflections) <--- Phase 9 (Denoising)
  |
  +---> Phase 11.1 (Adaptive DDGI Research)
  
Phase 9 (Denoising) <--- feeds into Phase 2 (DDGI temporal), Phase 3 (SSGI temporal),
                          Phase 4 (SSR temporal), Phase 8 (RT denoising)
```

---

## Effort Summary

| Phase | Tasks | Effort Estimate |
|-------|-------|-----------------|
| P1 Foundation | 5 | Small-Medium (2-3 weeks) |
| P2 DDGI Core | 9 | Medium-Large (6-8 weeks) |
| P3 SSGI | 2 | Medium (2 weeks) |
| P4 SSR Core | 5 | Medium-Large (4-5 weeks) |
| P5 Reflection Probes | 6 | Medium-Large (5-6 weeks) |
| P6 Planar Reflections | 2 | Medium (2 weeks) |
| P7 Voxel GI | 3 | Large (4-5 weeks) |
| P8 RT Reflections | 5 | Large (6-8 weeks) |
| P9 Denoising | 3 | Medium-Large (3-4 weeks) |
| P10 Visualization | 1 | Medium (1-2 weeks) |
| P11 Research | 3 | Large, variable (4-8 weeks research) |
| **Total** | **45** | **~39-53 weeks** |
