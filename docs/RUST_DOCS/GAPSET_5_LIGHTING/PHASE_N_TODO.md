# GAPSET_5_LIGHTING -- Task Inventory

> **Format**: T-LIT-{PHASE}.{N}
> **Total tasks**: 49
> **Legend**: [R] = Research task, [I] = Implementation task, [S] = Specification task

---

## Phase 1: GPU Light Data Infrastructure (5 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-1.1 | Build `light_types.rs` with LightTypeGPU enum and LightUnion tagged union | [I] | S15 math.rs | 3 days | All 7 light types defined as GPU-compatible repr(C) structs. Size verified via static_assert. |
| T-LIT-1.2 | Implement SoA buffer builder in `light_types.rs` — converts Vec<Light> into 7 SoA GPU buffers | [I] | T-LIT-1.1, S14 RHI | 4 days | Buffer upload produces correct byte arrays matching Python reference. Buffer sizes computed exactly. |
| T-LIT-1.3 | Build `lighting_system.py` orchestrator with dirty tracking via TrackedDescriptor | [I] | S15 component_store | 3 days | Orchestrator polls dirty components, rebuilds only changed lights. Full rebuild on add/remove. |
| T-LIT-1.4 | Implement CPU-to-GPU upload path with staging buffer ring | [I] | T-LIT-1.2, T-LIT-1.3, S14 RHI | 2 days | Light data uploaded to GPU correctly. Verified via GPU readback test. |
| T-LIT-1.5 | Build bind group layout for light data buffers (7 storage buffers + 2 uniform buffers for camera/grid config) | [I] | T-LIT-1.2, S14 RHI | 2 days | All bind groups created, validated via wgpu validation layers. |

---

## Phase 2: Froxel Clustered Culling Compute Shader (5 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-2.1 | Implement `light_culling.comp.wgsl` — per-froxel AABB reconstruction in view space | [I] | T-LIT-1.4 (light buffers) | 4 days | Froxel AABBs computed match Python FroxelGrid reference output within floating-point tolerance. |
| T-LIT-2.2 | Implement per-froxel light intersection tests (6 light type branches) in compute shader | [I] | T-LIT-2.1 | 5 days | Per-froxel light lists match Python ClusteredLightCuller reference. Directional in all froxels. Point/spot/IES correctly culled. |
| T-LIT-2.3 | Implement atomic index buffer compaction with overflow handling (MAX_LIGHTS_PER_FROXEL = 128) | [I] | T-LIT-2.2 | 3 days | Index buffer contiguous with correct offsets/counts. Atomic counter never exceeds allocated buffer size. Overflow froxels drop farthest lights. |
| T-LIT-2.4 | Build Rust dispatch in `culling.rs` — workgroup sizing, buffer allocation, dispatch dimensions | [I] | T-LIT-2.1, S14 RHI | 3 days | Compute pass dispatches correctly for all grid configurations (16x9x24, 12x8x16, 8x4x12). |
| T-LIT-2.5 | Implement GPU-readback correctness test vs. Python CPU reference (4 test scenes) | [I] | T-LIT-2.3, T-LIT-2.4 | 3 days | GPU froxel light lists match CPU reference in all 4 test scenes (simple, complex, overflow, empty). |

---

## Phase 3: Deferred PBR Lighting Compute Shader (7 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-3.1 | Implement shared `light_eval.wgsl` with attenuation functions, SurfaceData struct, BRDF dispatch | [I] | S3 material_shared.wgsl (or minimal local version) | 3 days | Attenuation functions match Python reference at >=10 distances per function. |
| T-LIT-3.2 | Implement directional + point + spot light evaluation functions in WGSL | [I] | T-LIT-3.1 | 4 days | Per-light output matches Python reference for random test cases (100 random configs per light type). |
| T-LIT-3.3 | Implement LTC LUT generation (`ltc_lut.py` or compute shader) for area lights | [I] | None (standalone) | 3 days | Generated LUT textures (64x64 RGBA16F + 64x64 R16F) produce correct area light integral for known reference BRDF parameters. |
| T-LIT-3.4 | Implement area light evaluation (RectArea + DiskArea) via LTC in WGSL | [I] | T-LIT-3.1, T-LIT-3.3 | 4 days | Area light energy matches reference LTC paper implementation (Heitz 2016) for 10 roughness x 5 cos_theta configurations. |
| T-LIT-3.5 | Implement IES light evaluation via 2D texture sampling + distance attenuation | [I] | T-LIT-3.1, S4-G4 (IES parser) | 2 days | IES light output matches Python IESProfile.sample() reference for 5 test profiles. |
| T-LIT-3.6 | Implement full `lighting_pass.comp.wgsl` — G-Buffer reading, froxel resolution, light accumulation, HDR output | [I] | T-LIT-2.4, T-LIT-3.2, T-LIT-3.4, T-LIT-3.5 | 5 days | Full deferred pass produces correct HDR output for 10-test-scene suite (compared against CPU reference rendering). |
| T-LIT-3.7 | Build Rust dispatch in `lighting_pass.rs` with bind group construction and workgroup sizing | [I] | T-LIT-3.6, S14 RHI | 3 days | Lighting pass dispatches with all 9+ bind groups correctly bound. Validation layers pass. |

---

## Phase 4: Cascaded Shadow Maps (6 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-4.1 | Implement PSSM cascade split computation (CPU) in `csm.rs` | [I] | S15 math | 2 days | Split distances match reference Python computation for n=0.1, f=1000, lambda=0..1, N=2..4. |
| T-LIT-4.2 | Implement frustum fitting + texel snapping (CPU) in `csm.rs` | [I] | T-LIT-4.1, S15 math | 3 days | Cascade projection matrices produce correct bounds (verified by rendering known points and checking coverage). Texel snap eliminates shimmer on camera motion. |
| T-LIT-4.3 | Write `shadow_csm.vert.wgsl` and `shadow_csm.frag.wgsl` for depth-only rendering | [I] | S14 RHI (depth texture + render pass) | 3 days | Depth rendered correctly for each cascade. Depth values in [0,1] with valid distribution. |
| T-LIT-4.4 | Implement cascade selection + blend in lighting pass shader | [I] | T-LIT-3.6, T-LIT-4.2 | 2 days | Cascade boundaries blend smoothly (visual test: no hard seam at cascade transition). Depth-based selection correct for all pixels within frustum. |
| T-LIT-4.5 | Build CSM atlas layout (2x2 grid) with per-cascade offset computation | [I] | T-LIT-4.3, T-LIT-6.1 (atlas) | 2 days | Each cascade renders into correct atlas quadrant. Sampling transform produces correct UV per cascade. |
| T-LIT-4.6 | Implement multi-viewport or sequential cascade rendering in Rust dispatch | [I] | T-LIT-4.3, S14 RHI | 3 days | All 4 cascades rendered per frame. Draw calls or viewport count verified. |

---

## Phase 5: Cube and Spot Shadow Maps (5 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-5.1 | Write `shadow_cube.vert.wgsl` and `shadow_cube.frag.wgsl` with 6-face depth rendering | [I] | S14 RHI (array texture + render pass) | 4 days | All 6 faces rendered with correct view transforms. Depth values valid. Faces cover full 360 degrees. |
| T-LIT-5.2 | Implement 2D array rendering strategy (6 layers) with hardware PCF sampling in Rust dispatch | [I] | T-LIT-5.1, S14 RHI | 3 days | Array texture created with 6 layers. Each layer corresponds to correct cubemap face. textSampleCompare works on array layers. |
| T-LIT-5.3 | Write `shadow_spot.vert.wgsl` and `shadow_spot.frag.wgsl` with perspective projection | [I] | S14 RHI | 2 days | Spot depth rendered within cone frustum. Valid depth distribution. |
| T-LIT-5.4 | Implement cone penumbra fade at spot shadow boundary | [I] | T-LIT-5.3, T-LIT-6.3 (shadow filter) | 1 day | Shadow smoothly fades at cone outer angle. No hard cutoff. |
| T-LIT-5.5 | Build cube + spot dispatch in `cube.rs` and `spot.rs` with per-light viewport setup | [I] | T-LIT-5.2, T-LIT-5.3 | 3 days | Per-light shadow passes dispatch correctly. Multiple point lights render into separate atlas tiles. |

---

## Phase 6: Shadow Atlas and Filtering (7 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-6.1 | Implement `atlas.rs` — 2D bin-packing allocator with 4 tile sizes (256-2048), free-list, overflow handling | [I] | None | 3 days | Allocation matches Python ShadowAtlas reference. Overflow degrades gracefully (reduces lowest-priority light). No fragmentation after 100-frame simulation. |
| T-LIT-6.2 | Build ShadowTileInfo GPU buffer from allocation results | [I] | T-LIT-6.1 | 2 days | Per-light tile metadata computed correctly (offset, scale, bounds). |
| T-LIT-6.3 | Implement `shadow_common.wgsl` with shared types (ShadowTileInfo), bias computation, and transform helpers | [I] | T-LIT-6.2 | 2 days | Common types usable by all shadow filter modules. Bias formulas match reference (slope-scaled + constant). |
| T-LIT-6.4 | Implement `shadow_filter_pcf.wgsl` with 4 kernel sizes (2x2, 3x3, 5x5, 7x7) and Poisson disk option | [I] | T-LIT-6.3 | 3 days | PCF output matches Python reference for all kernel sizes at 10+ test points. Hardware depth comparison used where available. |
| T-LIT-6.5 | Implement `shadow_filter_pcss.wgsl` with 3-step blocker search + penumbra + PCF | [I] | T-LIT-6.4 | 4 days | PCSS output produces wider penumbras with increasing occluder distance. Blocker search averaged depth matches reference. |
| T-LIT-6.6 | Implement `shadow_filter_vsm.wgsl` with Chebyshev inequality + configurable bleed reduction (exponent 8-64) | [I][R] | T-LIT-6.3 | 3 days | VSM pre-filter (separable Gaussian blur) produces smooth shadow maps. Bleeding reduction effective on thin occluders (no light leak at VSM_BLEED_REDUCTION=32+). Research note: recommend start exponent. |
| T-LIT-6.7 | Implement `shadow_filter_esm.wgsl` with configurable exponential constant `c` and pre-filter blur | [I][R] | T-LIT-6.3 | 3 days | ESM shadow comparison produces correct ratio. `c` parameter tunable. Research note: document quantization behavior at high `c` in 16-bit textures. |

---

## Phase 7: Virtual Shadow Maps (8 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-7.1 | Research: Survey published VSM implementations (UE5 GDC talks, academic papers) and derive page pool sizing equations | [R] | None | 5 days | Research document produced with: page pool size recommendations, clipmap level count justification, eviction policy comparison (LRU vs ticked LRU vs FIFO), and risk assessment. |
| T-LIT-7.2 | Research: Design feedback pass resolution, determine optimal fraction (1/4 vs 1/8 vs 1/16) with performance/quality analysis | [R] | T-LIT-7.1 | 3 days | Research document with resolution tradeoff analysis. Recommended default (1/8th) with justification. |
| T-LIT-7.3 | Research: Clipmap level transition strategy — evaluate dither, cross-fade, and hierarchical blend approaches | [R] | T-LIT-7.1 | 3 days | Research document with recommended transition method + WGSL pseudocode. |
| T-LIT-7.4 | Implement virtual address space + page table management in `vsm.rs` (1024x1024 page table, 16K virtual resolution) | [I] | T-LIT-7.1 | 4 days | Page table allocated with 4-byte entries. Virtual-to-physical mapping correct. NULL_PAGE entries handled. |
| T-LIT-7.5 | Implement `shadow_vsm_page.comp.wgsl` feedback pass — reads depth, computes virtual page coords, atomically writes page IDs | [I] | T-LIT-7.2, S1 frame graph | 5 days | Feedback pass produces correct page ID list for known test scene. Unique pages only (no duplicates in feedback buffer). |
| T-LIT-7.6 | Implement CPU readback loop with triple-buffered staging (ring of 3 resources) to avoid GPU stalls | [I] | T-LIT-7.5, S14 RHI | 4 days | Readback completes without stalling GPU pipe. 3-buffer ring provides continuous coverage. Max 1024 pages requested per frame. |
| T-LIT-7.7 | Implement page cache with LRU eviction policy (frame counter) or ticked LRU free list | [I] | T-LIT-7.1, T-LIT-7.6 | 4 days | Pages evicted after N=60 untouched frames (configurable). Cache hit/miss stats tracked. Miss rate <5% in test scenes. |
| T-LIT-7.8 | Implement VSM sampling in lighting pass with NULL_PAGE fallback (conservative shadow) | [I] | T-LIT-7.4, T-LIT-7.5, T-LIT-3.6 | 3 days | Unallocated pages render as shadowed. Page table lookups add <0.5% frame time overhead. |

---

## Phase 8: Contact Shadow Pass (3 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-8.1 | Implement `contact_shadow.comp.wgsl` — depth reconstruction, normal bias, screen-space ray march | [I] | S2 depth buffer, S14 RHI | 4 days | Contact shadows appear at surface intersections (character feet on ground). No self-shadowing on flat surfaces. |
| T-LIT-8.2 | Implement contact shadow blending with shadow map results (`final = min(shadow_map, contact_shadow)`) | [I] | T-LIT-8.1, T-LIT-3.6 | 1 day | Contact shadows fill in detail missed by shadow maps. No double-darkening. |
| T-LIT-8.3 | Build Rust dispatch in `contact.rs` with quality tier configuration (8/16/32/64 steps) | [I] | T-LIT-8.1, S14 RHI | 2 days | Contact pass dispatches with correct workgroup sizing. Step count configurable per quality tier. |

---

## Phase 9: Integration, Wiring, and Polish (8 tasks)

| TASK_ID | Description | Type | Dependencies | Effort | Acceptance Criteria |
|---------|-------------|------|--------------|--------|---------------------|
| T-LIT-9.1 | Wrap all compute + graphics passes into S1 frame graph nodes with explicit dependencies and barriers | [I] | S1 frame graph compiler, T-LIT-3.6, T-LIT-4.5, T-LIT-6.1 | 5 days | Frame graph compiles with correct pass ordering. Barriers inserted between dependent passes. No validation errors. |
| T-LIT-9.2 | Implement IES file parser (`ies_parser.py`) for LM-63-2002 format | [I] | None (standalone) | 2 days | Parser handles: keywords, tilt=NONE, lamp multiplier, units (lm/cd), vertical/horizontal angle counts, N*M candela values, symmetric/asymmetric angle handling, error on malformed files. |
| T-LIT-9.3 | Wire shadow request pipeline — collect @shadow_caster lights, forward to ShadowSystem for atlas allocation | [I] | T-LIT-1.3, T-LIT-6.1 | 3 days | Shadow requests gathered per frame. Atlas allocation performed before shadow rendering. Per-light ShadowTileInfo correctly populated. |
| T-LIT-9.4 | Wire GI light list handoff — build per-frame list of GI-contributing lights for S6 | [I] | T-LIT-1.3 | 2 days | GI light list contains all lights with @gi_contributor decorator. Importance levels correctly mapped. Handoff buffer format matches S6 interface. |
| T-LIT-9.5 | Specify and implement separate cast/receive shadow flags per object | [S][I] | T-LIT-6.1 | 2 days | Objects can cast without receiving (e.g., background geometry) and receive without casting. Flag propagates through shadow pipeline. |
| T-LIT-9.6 | Implement shadow LOD — distance-based resolution reduction for distant casters | [S][I] | T-LIT-6.1 | 3 days | Distant shadow casters automatically get lower resolution atlas tiles. LOD transition distances configurable per scene. |
| T-LIT-9.7 | Implement multiple directional light shadow support with atlas sharing | [S][I] | T-LIT-4.5, T-LIT-6.1 | 3 days | Two+ directional lights (sun + moon) each get separate CSM atlas regions. Atlas allocation handles multiple directionals. |
| T-LIT-9.8 | Implement shadow color/density modulation for stylized rendering + adaptive slope-scaled bias | [I] | T-LIT-6.3 | 2 days | Shadow density parameter (0-1) modulates shadow darkness. Slope-scaled bias computed per-pixel from depth derivatives. |

---

## Summary

| Phase | Tasks | Total Effort (days) |
|-------|-------|---------------------|
| Phase 1: GPU Light Data Infrastructure | 5 | 14 |
| Phase 2: Froxel Clustered Culling | 5 | 18 |
| Phase 3: Deferred PBR Lighting | 7 | 24 |
| Phase 4: Cascaded Shadow Maps | 6 | 15 |
| Phase 5: Cube + Spot Shadow Maps | 5 | 13 |
| Phase 6: Shadow Atlas + Filtering | 7 | 20 |
| Phase 7: Virtual Shadow Maps | 8 | 31 |
| Phase 8: Contact Shadow Pass | 3 | 7 |
| Phase 9: Integration + Polish | 8 | 22 |
| **Total** | **49** | **~164** |

Research tasks: 4 (T-LIT-7.1 through T-LIT-7.3)
Implementation tasks: 43
Specification tasks: 3 (T-LIT-9.5, T-LIT-9.6, T-LIT-9.7)

### Gap Coverage

| Gap ID | Covered By | Severity |
|--------|-----------|----------|
| S4-G1 | T-LIT-2.x, T-LIT-3.x | CRITICAL |
| S4-G2 | T-LIT-2.2 (optimization note) | LOW |
| S4-G3 | T-LIT-3.4 (area light normalization) | LOW |
| S4-G4 | T-LIT-9.2 | MEDIUM |
| S4-G5 | T-LIT-3.3 | HIGH |
| S4-G6 | T-LIT-1.x | CRITICAL |
| S4-G7 | T-LIT-9.3 | HIGH |
| S4-G8 | T-LIT-9.4 | MEDIUM |
| S5-G1 | T-LIT-4.x, T-LIT-5.x, T-LIT-6.x, T-LIT-8.x | CRITICAL |
| S5-G2 | T-LIT-6.7 | HIGH |
| S5-G3 | T-LIT-6.6 (research + impl) | HIGH |
| S5-G4 | T-LIT-7.x (research + impl) | HIGH |
| S5-G5 | T-LIT-9.5 | MEDIUM |
| S5-G6 | T-LIT-9.6 | MEDIUM |
| S5-G7 | T-LIT-9.7 | MEDIUM |
| S5-G8 | T-LIT-9.8 | LOW |
| S5-G9 | T-LIT-6.5 (PCSS design) | MEDIUM |
| S5-G10 | T-LIT-7.3 (clipmap research) | MEDIUM |
| S5-G11 | T-LIT-9.3 (static invalidation) | MEDIUM |
| S5-G12 | T-LIT-9.1 (per-light filter API) | MEDIUM |
| S5-G13 | (Deferred) | LOW |
| S5-G14 | (Deferred) | LOW |
| S5-G15 | T-LIT-9.8 | MEDIUM |
| S5-G16 | T-LIT-9.5 | MEDIUM |
| S5-G17 | T-LIT-9.6 | LOW |
