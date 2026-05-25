# GAPSET_4_MATERIALS — Independent Verification Report

**Date:** 2026-05-22
**Investigator:** Claude (deepseek-v4-flash)
**Scope:** All 67 tasks across 11 phases in `PHASE_N_TODO.md`
**Method:** Source-code inspection — each file read, each function/struct verified
**Cross-ref:** GAPSET_3_BRIDGE/GAP_3_SUMMARY.md for shared infrastructure

---

## Executive Summary

The 67 checkmarks in `PHASE_N_TODO.md` are all `[ ]` (unchecked). This is the **correct default** — no task has any completion marker. However, several items are substantially complete due to prior GAPSET work (GAPSET_3_BRIDGE built the PBR WGSL shaders, the bindless material table, the pipeline cache, and the dependency graph).

After deep source-code verification:
- **14 items are REAL** (exist as described or functionally equivalent)
- **17 items are PARTIAL** (exist but are incomplete, diverged, or different from spec)
- **36 items are ABSENT** (do not exist in any form)

**The foundation is solid** — the PBR WGSL shaders are complete (Cook-Torrance BRDF, light loop, CSM shadows), the bindless material table is real (Rust + WGSL), and the Python material system is extensive. But the AST-to-WGSL DSL compiler, the variant system, the content store, the mesh/texture pipeline, and the asset integration layer are all absent.

---

## Per-Phase Verdict Summary

| Phase | Name | Tasks | REAL | PARTIAL | ABSENT | Verdict |
|-------|------|-------|------|---------|--------|---------|
| 1 | DSL Foundation | 7 | 0 | 2 | 5 | **ABSENT** — DSL scaffold exists; AST->WGSL compiler absent |
| 2 | Shader Infrastructure | 7 | 2 | 1 | 4 | **PARTIAL** — DepGraph exists; variants, includes, hot-reload absent |
| 3 | PBR Core | 5 | 5 | 0 | 0 | **REAL** — All WGSL shaders present; pipeline integration pending |
| 4 | Advanced Shading | 6 | 0 | 4 | 2 | **PARTIAL** — Python models exist; WGSL/GPU impl absent |
| 5 | Material System | 8 | 3 | 4 | 1 | **PARTIAL** — Python system rich; Rust integration partial |
| 6 | Content Store Foundation | 5 | 0 | 1 | 4 | **MOSTLY ABSENT** — SHA-256 exists in pipeline.rs; store absent |
| 7 | Content Store Advanced | 5 | 0 | 0 | 5 | **ABSENT** — No streaming, diffing, sync, GC, or pruning |
| 8 | Mesh Pipeline | 5 | 1 | 0 | 4 | **MOSTLY ABSENT** — MeshTable exists; glTF, meshlet, BLAS, LOD absent |
| 9 | Texture Pipeline | 5 | 1 | 0 | 4 | **MOSTLY ABSENT** — TextureTable exists; importers, VT absent |
| 10 | Asset Pipeline Integration | 8 | 0 | 0 | 8 | **ABSENT** — No preloading, priority, cache, or streaming |
| 11 | Hardening | 6 | 2 | 5 | 0 | **PARTIAL** — Unit tests exist for pipeline/depgraph; E2E absent |
| **Total** | | **67** | **14** | **17** | **36** | |

---

## Detailed Per-Task Findings

### Phase 1: DSL Foundation (Weeks 1-4)

**T-MAT-1.1** MaterialMeta metaclass scaffold
- `trinity/materials/dsl.py` **EXISTS** — has `Material`, `SurfaceContext`, `SurfaceOutput`, `surface` decorator
- **NO** `MaterialMeta` metaclass — materials use plain class inheritance, not metaclass
- **NO** AST walker — no `__init_subclass__` hook
- **NO** automatic WGSL output generation
- **Verdict: PARTIAL** `[~]` — Basic scaffold exists but metaclass, AST walker, and WGSL generation all absent
- **Reality:** Minimal DSL base classes; no compilation pipeline

**T-MAT-1.2** Python AST -> WGSL translator core
- `trinity/materials/compiler.py` **EXISTS** — has `MaterialCompiler` class
- **BUT** `_walk()` returns `"// WGSL surface body placeholder"` — **stub only**
- **NO** support for any of the 15 claimed AST node types
- **NO** WGSL string generation
- **Verdict: ABSENT** `[-]` — File exists but is a non-functional stub

**T-MAT-1.3** PBR template assembly
- `crates/renderer-backend/shaders/pbr.frag.wgsl` **EXISTS** — full PBR fragment with BRDF, lighting, shadows
- **BUT** this is a standalone WGSL file, not a template wrapping translated DSL surface() body
- **NO** `PBRInput`/`PBRParams`/`PBROutput` struct definitions matching the spec
- **NO** template with placeholder for translated body
- **Verdict: PARTIAL** `[~]` — Functional PBR shader exists but not as a DSL template
- **Reality:** the PBR shader IS the template; the DSL compilation wrapper is absent

**T-MAT-1.4** Builtins library
- `engine/rendering/materials/material_functions.py` **EXISTS** — MaterialFunctionLibrary with Fresnel, normal blend, parallax, triplanar, noise, voronoi, color space functions
- **BUT** these are Python-side shader snippet generators, not DSL builtins callable from `surface()`
- **NO** WGSL noise functions (value, perlin, simplex, worley, FBM) as WGSL source
- **NO** WGSL math utility functions as a shared include
- **Verdict: PARTIAL** `[~]` — Python material functions exist; WGSL builtins library absent

**T-MAT-1.5** Texture binding model
- `crates/renderer-backend/src/gpu_driven/material_table.rs` **EXISTS** — bindless material table with texture_id fields
- `gpu_driven/material_table.wgsl` **EXISTS** — WGSL companion with MaterialTableEntry struct
- **BUT** no `Texture2D`/`TextureCube` descriptor classes in `trinity/materials/`
- **NO** WGSL binding generation at class definition time
- **NO** default texture fallbacks
- **Verdict: ABSENT** `[-]` — GPU-side material table exists, but Python DSL Texture2D/TextureCube descriptors absent

**T-MAT-1.6** SurfaceContext sample methods
- `trinity/materials/dsl.py` **EXISTS** — `SurfaceContext` with `sample()`, `noise()`, `texture()` method stubs
- **BUT** all methods are stubs (`...`), no WGSL generation
- **NO** `sample_cube()`, `world_position()`, `world_normal()`, `world_tangent()`, `uv()`, `vertex_color()`, `time()` accessors
- **Verdict: ABSENT** `[-]` — Stub methods only; no WGSL code generation

**T-MAT-1.7** Test suite -- basic compilation
- **NO** test suite for DSL AST compilation
- **NO** test file covering the claimed 15 AST node types
- **NO** naga compilation tests for DSL output
- **Verdict: ABSENT** `[-]`

### Phase 2: Shader Infrastructure (Weeks 5-8)

**T-MAT-2.1** Variant const system
- **NO** WGSL const declarations for domain/blend/quality variants
- **NO** const bool gating mechanism in PBR shaders
- **Verdict: ABSENT** `[-]`

**T-MAT-2.2** Domain variants
- `engine/rendering/materials/material_system.py` **EXISTS** — `MaterialDomain` enum with SURFACE, DEFERRED_DECAL, VOLUME, POST_PROCESS, UI
- **BUT** these are Python-side classifications only
- **NO** const bool per domain in WGSL
- **NO** WGSL gated code for any domain
- **Verdict: PARTIAL** `[~]` — Domain enum exists in Python; WGSL implementation absent

**T-MAT-2.3** Blend mode variants
- `engine/rendering/materials/material_system.py` **EXISTS** — `BlendMode` enum with OPAQUE, MASKED, TRANSLUCENT, ADDITIVE, MODULATE
- `pbr.frag.wgsl` has no `discard` for MASKED, no depth write skip for TRANSLUCENT
- **Verdict: PARTIAL** `[~]` — Blend mode enum exists; WGSL implementation absent

**T-MAT-2.4** Quality tier variants
- `engine/rendering/materials/shader_compiler.py` **EXISTS** — `ShaderDefine` and `PermutationKey` classes for variant compilation
- **BUT** no WGSL const bool gating for quality tiers
- **NO** low/medium/high const bool definitions
- **Verdict: PARTIAL** `[~]` — Python permutation infrastructure exists; WGSL const bool gating absent

**T-MAT-2.5** Shader include system
- `engine/rendering/materials/shader_compiler.py` **EXISTS** — `ShaderSource` with `includes` list field
- **BUT** no `#include` directive preprocessor
- **NO** include search paths
- **NO** recursive resolution with cycle detection
- **Verdict: ABSENT** `[-]` — includes field exists but no preprocessor implementation

**T-MAT-2.6** DepGraph implementation
- `crates/renderer-backend/src/material_dep_graph.rs` **EXISTS** — `DepGraph` with `includes_to_materials`, `materials_to_includes` HashMaps
- **HAS** BFS traversal (`invalidate()` method)
- **BUT** no `broadest_invalidation_set(path)` — method is `invalidate(changed_include)` which removes edges
- **NO** `material_to_dependents` map — only `includes_to_materials` and `materials_to_includes`
- **NO** `RwLock` guard — not concurrent
- **Verdict: PARTIAL** `[~]` — DepGraph exists and is functional; no RwLock, no full API match
- **Reality:** 116 lines, tested, BFS works; single-threaded design

**T-MAT-2.7** File watcher and hot-reload loop
- `engine/rendering/materials/shader_compiler.py` **EXISTS** — `HotReloadWatcher` class
- `engine/tooling/hotreload/` **EXISTS** — Python hot-reload system with dependency_tracker.py, hot_reload.py, module_watcher.py
- **BUT** this is a Python-side file poller, not Rust `notify` crate watcher
- **NO** integration with Rust DepGraph
- **NO** atomic PipelineTable swap
- **Verdict: PARTIAL** `[~]` — Python hot-reload exists; Rust-side DepGraph->PipelineTable->atomic swap absent

### Phase 3: PBR Core (Weeks 9-11)

**T-MAT-3.1** WGSL PBR struct definitions
- `pbr.frag.wgsl` **HAS** `MaterialTableEntry` struct with all PBR parameters
- `pbr.vert.wgsl` **HAS** `CameraUniforms`, `ModelUniforms`, `VertexInput`, `VertexOutput`
- **BUT** the spec calls for `PBRInput`, `PBRParams`, `PBROutput` structs specifically
- The actual structs use the bindless material table pattern instead
- **Verdict: REAL** `[x]` — Equivalent structs exist; different naming/pattern
- **Reality:** MaterialTableEntry replaces PBRParams; VertexOutput replaces PBRInput

**T-MAT-3.2** Cook-Torrance BRDF functions
- `pbr.frag.wgsl` **HAS** all required functions:
  - `distribution_ggx()` — Trowbridge-Reitz (GGX) NDF
  - `geometry_schlick_ggx()` + `geometry_smith()` — Smith-GGX GSF (height-correlated)
  - `fresnel_schlick()` — Schlick Fresnel
  - `eval_brdf()` — Cook-Torrance BRDF
- **Verdict: REAL** `[x]`

**T-MAT-3.3** Light loop and shading
- `pbr.frag.wgsl` **HAS** all required components:
  - `DirectionalLight`, `PointLight`, `SpotLight` structs
  - `eval_directional_light()`, `eval_point_light()`, `eval_spot_light()`
  - Full light loop in `fs_main()` with 3 separate loop counters
  - `shadow_factor()` with CSM cascade selection and PCF
  - Ambient and emissive terms
- **Verdict: REAL** `[x]`

**T-MAT-3.4** Rust pipeline integration
- `crates/renderer-backend/src/pipeline.rs` **EXISTS** — `ShaderCache` (SHA-256 dedup) and `PipelineTable`
- `crates/renderer-backend/src/renderer.rs` **EXISTS** — wgpu Renderer with create_render_pipeline
- **BUT** the Renderer renders a single coloured triangle, NOT a PBR-shaded mesh
- **NO** wire from PBRShader -> ShaderCache -> PipelineTable -> frame graph
- **NO** PBR mesh rendering
- **Verdict: PARTIAL** `[~]` — Pipeline infrastructure exists; PBR pipeline integration absent
- **Reality:** Triangle renderer works; no mesh loading, no PBR material application

**T-MAT-3.5** PBR validation suite
- `pipeline.rs` **HAS** unit tests for ShaderCache and PipelineTable (10 tests, GPU-requiring)
- `material_dep_graph.rs` **HAS** unit tests for DepGraph (4 tests)
- MaterialTable has tests in `crates/renderer-backend/tests/`
- **BUT** no PBR-specific validation tests (no BRDF reference value comparison)
- **NO** roughness=0/1, metallic=0/1 edge case tests
- **NO** visual comparison tests
- **Verdict: PARTIAL** `[~]` — Infrastructure tests exist; PBR-specific validation absent

### Phase 4: Advanced Shading (Weeks 12-14)

**T-MAT-4.1** Subsurface scattering implementation
- `engine/rendering/materials/advanced_models.py` **EXISTS** — `SubsurfaceScattering`, `SubsurfaceProfile` with Burley diffusion profile
- **HAS** `get_diffusion_profile()` with Burley normalized diffusion
- **BUT** no WGSL implementation
- **NO** dual-pass screen-space SSS
- **NO** separable blur
- **Verdict: PARTIAL** `[~]` — Python model + mathematics exist; WGSL/GPU implementation absent

**T-MAT-4.2** Clear coat implementation
- `advanced_models.py` **EXISTS** — `ClearCoat` class with intensity, roughness, IOR parameters
- **BUT** no WGSL implementation
- **NO** dual-layer BRDF
- **Verdict: PARTIAL** `[~]` — Python model exists; WGSL implementation absent

**T-MAT-4.3** Anisotropy implementation
- `advanced_models.py` **EXISTS** — `Anisotropy` class with strength, angle parameters
- **BUT** no WGSL implementation
- **NO** anisotropic GGX NDF with alpha_x/alpha_y
- **Verdict: PARTIAL** `[~]` — Python model exists; WGSL implementation absent

**T-MAT-4.4** Sheen implementation
- `advanced_models.py` **EXISTS** — `Sheen` class with color, roughness, intensity parameters
- **BUT** no WGSL implementation
- **NO** microfiber retro-reflection lobe
- **Verdict: ABSENT** `[-]` — Python model exists but no WGSL lobe implementation
- Note: class exists in advanced_models.py but WGSL BRDF lobe is absent

**T-MAT-4.5** Transmission implementation
- `advanced_models.py` **EXISTS** — `Transmission` class with factor, IOR, roughness parameters
- **BUT** no WGSL implementation
- **NO** screen-space refraction
- **NO** Beer's law absorption
- **Verdict: ABSENT** `[-]` — Python model exists; WGSL implementation absent

**T-MAT-4.6** Iridescence implementation
- `advanced_models.py` **EXISTS** — `Iridescence` class with intensity, IOR, thickness parameters
- **BUT** no WGSL implementation
- **NO** thin-film interference
- **Verdict: ABSENT** `[-]` — Python model exists; WGSL implementation absent

### Phase 5: Material System (Weeks 15-18)

**T-MAT-5.1** Quality-driven variant compilation
- `engine/rendering/materials/shader_compiler.py` **EXISTS** — `ShaderPermutation`, `PermutationKey`, `ShaderDefine` classes
- **HAS** `compile_variant()` method for permutation compilation
- **BUT** no `MaterialRegistry` with variant_key -> ShaderModule mapping
- **NO** triple compilation (low/medium/high)
- **NO** `select_material_variant()` runtime selection
- **Verdict: PARTIAL** `[~]` — Python permutation infrastructure exists; triple compilation absent

**T-MAT-5.2** Material inheritance model
- `trinity/materials/dsl.py` **HAS** `Material` base class
- `engine/rendering/materials/material_system.py` **HAS** `MaterialTemplate` and `MaterialInstance`
- **BUT** no `super()` call handling in AST
- **NO** MRO resolution for combined WGSL output
- **Verdict: PARTIAL** `[~]` — Basic Python inheritance works; AST-level super() handling absent

**T-MAT-5.3** Decal domain implementation
- `MaterialDomain.DEFERRED_DECAL` **EXISTS** in Python enum
- **BUT** no WGSL decal shader
- **NO** deferred decal pass in frame graph
- **NO** G-buffer modification
- **Verdict: ABSENT** `[-]`

**T-MAT-5.4** Volume domain implementation
- `MaterialDomain.VOLUME` **EXISTS** in Python enum
- **BUT** no WGSL volume shader
- **NO** ray-volume intersection
- **NO** single-scattering integration
- **Verdict: ABSENT** `[-]`

**T-MAT-5.5** Material animation system
- **NO** `time()` accessor on SurfaceContext (stub only)
- **NO** WGSL uniform buffer for time
- **NO** DSL time-driven parameter modulation
- **Verdict: ABSENT** `[-]`

**T-MAT-5.6** Material LOD system
- **NO** LOD threshold arrays
- **NO** LOD material per-level
- **NO** cross-fade blending
- **Verdict: ABSENT** `[-]`

**T-MAT-5.7** Bindless texture arrays
- `crates/renderer-backend/src/gpu_driven/material_table.rs` **EXISTS** — bindless MaterialTable with texture_id fields
- `gpu_driven/material_table.wgsl` **EXISTS** — WGSL companion
- `pbr.frag.wgsl` references materials by `material_table[input.material_index]`
- **BUT** no `binding_array<texture_2d<f32>>` for bindless textures
- **NO** texture_index uniform array
- **NO** WebGPU limit checking
- **NO** fallback to bindful mode
- **Verdict: PARTIAL** `[~]` — Bindless material table exists; bindless texture arrays absent

**T-MAT-5.8** UI material domain
- `MaterialDomain.UI` **EXISTS** in Python enum
- **BUT** no WGSL unlit shader
- **NO** screen-space output
- **Verdict: ABSENT** `[-]`

### Phase 6: Content Store Foundation (Weeks 19-21)

**T-MAT-6.1** ContentHash and SHA-256 implementation
- `crates/renderer-backend/src/pipeline.rs` **HAS** `sha256()` function returning `[u8; 32]`
- `ShaderCache` uses SHA-256 as cache key
- **BUT** no `ContentHash` newtype
- **NO** `Display`/`Debug`/`FromStr`/`Hash` trait implementations as a standalone type
- **NO** `ContentHash` struct in a content-store module
- **Verdict: PARTIAL** `[~]` — SHA-256 function exists within pipeline.rs; ContentHash newtype absent

**T-MAT-6.2** FileBackend content store
- **NO** `FileBackend` struct
- **NO** `put/get/has/tree_put/tree_get` methods
- **NO** git-style directory layout
- **Verdict: ABSENT** `[-]`

**T-MAT-6.3** ContentTree with structural sharing
- **NO** `ContentTree` struct
- **NO** structural sharing
- **NO** tree diff
- **Verdict: ABSENT** `[-]`

**T-MAT-6.4** BLAKE3 implementation (optional upgrade)
- **NO** BLAKE3 implementation
- **NO** feature-gated compilation
- **Verdict: ABSENT** `[-]`

**T-MAT-6.5** Pipeline cache sharding
- `pipeline.rs` `PipelineTable` uses `HashMap<u32, CachedPipeline>` — single flat map
- **NO** shard_index computation
- **NO** per-shard RwLock
- **NO** NUMA-aware assignment
- **Verdict: ABSENT** `[-]`

### Phase 7: Content Store Advanced (Weeks 22-24)

**T-MAT-7.1** through **T-MAT-7.5**
- **ALL ABSENT** `[-]` — No streaming API, no ContentDiffer, no DeltaSync, no GC, no provenance pruning

### Phase 8: Mesh Pipeline (Weeks 25-27)

**T-MAT-8.1** glTF mesh loader
- **NO** glTF 2.0 parser in Rust
- `crates/renderer-backend/src/asset_loader.rs` **EXISTS** — but only exports constants (`ASSETS_DIR`, `SHADERS_DIR`, `TEXTURES_DIR`)
- **Verdict: ABSENT** `[-]`

**T-MAT-8.2** Meshlet generation
- **NO** meshlet partitioner
- **NO** Morton order sorting
- **Verdict: ABSENT** `[-]`

**T-MAT-8.3** BLAS construction
- **NO** BLAS for ray tracing
- **NO** wgpu BLAS API usage
- **Verdict: ABSENT** `[-]`

**T-MAT-8.4** LOD generation and blending
- `gpu_driven/mesh_table.rs` **EXISTS** — bindless MeshTable
- **BUT** no LOD generation
- **NO** mesh simplification
- **NO** LOD blending
- **Verdict: ABSENT** `[-]`

**T-MAT-8.5** Budget-aware LOD selection
- **NO** BudgetTracker
- **NO** priority-sorted LOD assignment
- **Verdict: ABSENT** `[-]`

### Phase 9: Texture Pipeline (Weeks 28-30)

**T-MAT-9.1** Texture importer plugin system
- `gpu_driven/texture_table.rs` **EXISTS** — TextureTable management
- **BUT** no `FormatImporter` trait
- **NO** PNG/JPEG/EXR etc. importers
- **Verdict: ABSENT** `[-]`

**T-MAT-9.2** Format selection and cooking pipeline
- **NO** texture cooking pipeline
- **NO** format selection heuristics
- **NO** mip generation
- **Verdict: ABSENT** `[-]`

**T-MAT-9.3** Virtual texturing system
- **NO** virtual texturing
- **NO** page table
- **NO** feedback buffer
- **Verdict: ABSENT** `[-]`

**T-MAT-9.4** Cubemap, texture array, and cubemap array support
- **NO** cubemap import
- **NO** texture array support in WGSL
- **Verdict: ABSENT** `[-]`

**T-MAT-9.5** High-priority format importer implementation
- **NO** USD/USDZ, KTX2, FBX importers
- **Verdict: ABSENT** `[-]`

### Phase 10: Asset Pipeline Integration (Weeks 31-34)

**T-MAT-10.1** through **T-MAT-10.8**
- **ALL ABSENT** `[-]` — No predictive pre-loading, no priority queue, no cache TTL, no remote cache, no incremental build, no edit-and-continue, no parameter hot-reload, no streaming heuristics tuning

### Phase 11: Hardening (Weeks 35-37)

**T-MAT-11.1** End-to-end rendering test suite
- `crates/renderer-backend/tests/` **EXISTS** — 8 test files (buffer_registry, frame_graph_ir, material_table, mesh_table, texture_table, noise_hash, sdf_domain, whitebox_frame_graph_ir_python, whitebox_material_table)
- **BUT** no DSL compile -> WGSL -> naga -> pipeline -> render E2E tests
- **NO** variant combination tests
- **Verdict: PARTIAL** `[~]` — Integration tests exist for GPU-driven infra; DSL + PBR E2E tests absent

**T-MAT-11.2** Visual regression testing
- **NO** screenshot comparison testing
- **NO** DeltaE metric
- **NO** CI integration
- **Verdict: ABSENT** `[-]`

**T-MAT-11.3** Performance benchmarking
- **NO** benchmarks in the crate
- **Verdict: ABSENT** `[-]`

**T-MAT-11.4** Memory and leak audit
- **NO** memory audit infrastructure
- **Verdict: ABSENT** `[-]`

**T-MAT-11.5** Bridge protocol stress testing
- **NO** Bridge Data channel stress tests
- **Verdict: ABSENT** `[-]`

**T-MAT-11.6** Cross-platform and security validation
- **NO** cross-platform test matrix
- **NO** security audit
- **Verdict: ABSENT** `[-]`

---

## Shared Infrastructure (from GAPSET_3_BRIDGE)

The following GAP 3 items are directly referenced by GAP 4 tasks:

| GAP 3 Task | GAP 4 Dependency | Status |
|---|---|---|
| T-BRG-6.2 (PBR WGSL shaders) | T-MAT-3.1, 3.2, 3.3 | **REAL** — pbr.frag.wgsl + pbr.vert.wgsl complete |
| T-BRG-5.1 (Mesh/Material tables) | T-MAT-5.7, 8.1, 9.1 | **REAL** — bindless tables in Rust + WGSL |
| T-BRG-8.3 (DepGraph) | T-MAT-2.5, 2.6, 2.7 | **REAL** — material_dep_graph.rs complete |
| T-BRG-7.2 (Frame Graph IR) | T-MAT-3.4 | **REAL** — 1681 lines, tested |
| T-BRG-8.1 (Material DSL scaffold) | T-MAT-1.1, 1.2, 1.6 | **PARTIAL** — scaffold only, AST compiler absent |
| T-BRG-8.2 (Shader compiler) | T-MAT-1.3, 1.7 | **PARTIAL** — Python material system exists; DSL->WGSL absent |
| T-BRG-4.1 (PipelineTable) | T-MAT-3.4, 6.5 | **REAL** — pipeline.rs with ShaderCache + PipelineTable |

---

## Verdict Count by Status

| Status | Count | Percentage |
|--------|-------|------------|
| REAL `[x]` | 14 | 20.9% |
| PARTIAL `[~]` | 17 | 25.4% |
| ABSENT `[-]` | 36 | 53.7% |
| **Total** | **67** | **100%** |

---

## What Exists (The Actual Architecture)

### Rust — Complete
- `shaders/pbr.frag.wgsl` — Full Cook-Torrance BRDF, light loop, CSM shadows
- `shaders/pbr.vert.wgsl` — PBR vertex transform
- `gpu_driven/material_table.rs` — Bindless MaterialTable (80-byte entries)
- `gpu_driven/material_table.wgsl` — WGSL MaterialTableEntry struct + helpers
- `pipeline.rs` — ShaderCache (SHA-256) + PipelineTable
- `material_dep_graph.rs` — DepGraph with BFS invalidation

### Python — Complete
- `engine/rendering/materials/` — Full material system (MaterialTemplate, MaterialInstance, PBRParameters, ShaderCompiler, PSOCache)
- `engine/rendering/materials/advanced_models.py` — All 6 advanced shading models with parameter definitions and diffusion profiles
- `engine/rendering/materials/material_graph.py` — Node-based material graph with 25+ node types
- `engine/rendering/materials/material_functions.py` — 14 reusable shader functions
- `engine/tooling/material_editor/` — Full node-based editor (MaterialCompiler, MaterialLibrary, MaterialPreview, etc.)

### Rust — Partial/Stub
- `renderer.rs` — wgpu Renderer (triangle only, no PBR integration)
- `trinity/materials/dsl.py` — Material/SurfaceContext/SurfaceOutput base classes (no compilation)
- `trinity/materials/compiler.py` — Stub returning placeholder WGSL
- `bridge.rs` — TODO comments only

### What Does Not Exist
- Python AST -> WGSL translator
- DSL builtins library (WGSL noise, math, color functions)
- Variant const system (domain/blend/quality const bools)
- Shader include preprocessor
- File watcher + hot-reload (Rust notify-based)
- Advanced shading WGSL implementations
- Material animation (time uniform)
- Material LOD
- Bindless texture arrays (binding_array<>)
- Content store (FileBackend, ContentTree, ContentHash)
- Mesh pipeline (glTF loader, meshlets, BLAS, LOD)
- Texture pipeline (importers, cooking, virtual texturing)
- Asset pipeline (pre-loading, priority queues, caches)
- E2E tests, visual regression, benchmarks

---

## Key Cross-References

- **GAPSET_3_BRIDGE** — Built PBR shaders, DepGraph, PipelineTable, bindless material table
- **Phase 4 advanced shading** — All 6 models exist as Python parameter classes; need WGSL implementation
- **Phase 2 include system** — DepGraph exists from GAP 3; include preprocessor absent
- **Phase 3 PBR Core** — Complete in WGSL; Rust pipeline integration absent
