# GAPSET_4_MATERIALS: Task Breakdown

Task ID format: `T-MAT-{PHASE}.{N}` where PHASE = 1-11 and N = sequential task number.

---

## Phase 1: DSL Foundation (Weeks 1-4)

Gaps: S3-G1, S3-G2, S3-G7, S3-G17

### T-MAT-1.1: MaterialMeta metaclass scaffold
- **Gap:** S3-G1 (CRITICAL)
- **Effort:** 2 days
- **Description:** Implement `MaterialMeta.__init_subclass__` in `dsl.py`. Hook class creation, extract `surface()` method source, invoke AST walker, collect WGSL output, register with pipeline cache. Implement `SurfaceContext` and `SurfaceOutput` Python proxy classes with type-annotated fields.
- **Acceptance:** Creating a class with `metaclass=MaterialMeta` and a `surface()` method produces valid WGSL string. SurfaceContext/SurfaceOutput can be instantiated with Python values.
- **Dependencies:** None

### T-MAT-1.2: Python AST -> WGSL translator core
- **Gap:** S3-G2 (CRITICAL)
- **Effort:** 3 days
- **Description:** Implement `PythonToWGSLTranslator` in `compiler.py`. Support the 15 core AST node types: Expr, Assign, AnnAssign, Call, BinOp, If, Attribute, Name, Constant, Subscript, UnaryOp, Compare, BoolOp, Return, Expr. Each node maps to a WGSL equivalent string.
- **Acceptance:** All 15 node types produce valid WGSL. Test inputs exercise each node type at least once.
- **Dependencies:** T-MAT-1.1

### T-MAT-1.3: PBR template assembly
- **Gap:** S3-G2 (CRITICAL)
- **Effort:** 2 days
- **Description:** Implement WGSL PBR template that wraps translated surface() body. Template includes: PBRInput/PBRParams/PBROutput struct definitions, vertex shader entry point, fragment shader entry point, BRDF function declarations, light loop scaffolding. Translated body is inserted into fragment main(). Compile via naga parse-validate.
- **Acceptance:** Template + translated body compiles through naga without errors. Template has placeholders for all PBR functions.
- **Dependencies:** T-MAT-1.2

### T-MAT-1.4: Builtins library
- **Gap:** S3-G7 (HIGH)
- **Effort:** 2 days
- **Description:** Implement `builtins.py` with noise functions (value, perlin, simplex, worley, FBM), math utilities (lerp, smoothstep, normalize, reflect, refract, clamp, saturate, mix), and color conversion functions (rgb_to_hsv, hsv_to_rgb, linear_to_srgb, srgb_to_linear, tonemap). Each function emits WGSL code when called from DSL surface() body.
- **Acceptance:** All builtins can be called from DSL surface() body. Generated WGSL compiles. Test verifies output correctness.
- **Dependencies:** T-MAT-1.2

### T-MAT-1.5: Texture binding model
- **Gap:** S3-G17 (HIGH)
- **Effort:** 2 days
- **Description:** Implement `textures.py` with Texture2D and TextureCube descriptor classes. Each descriptor generates WGSL bindings (texture + sampler declarations) at class definition time. Support srgb flag for automatic format conversion. Support default texture fallbacks (white, flat_normal, black, etc.).
- **Acceptance:** Declaring `albedo = Texture2D(default="white", srgb=True)` in a MaterialMeta class generates correct WGSL bindings.
- **Dependencies:** T-MAT-1.1

### T-MAT-1.6: SurfaceContext sample methods
- **Gap:** S3-G1, S3-G17 (CRITICAL + HIGH)
- **Effort:** 1 day
- **Description:** Implement sample() and sample_cube() methods on SurfaceContext that map to WGSL textureSample calls. Also implement world_position(), world_normal(), world_tangent(), uv(), vertex_color(), time() accessors that map to PBRInput struct field reads.
- **Acceptance:** ctx.sample(texture, uv()) generates correct WGSL textureSample call. All accessors compile.
- **Dependencies:** T-MAT-1.2, T-MAT-1.5

### T-MAT-1.7: Test suite -- basic compilation
- **Gap:** S3-G1, S3-G2 (CRITICAL)
- **Effort:** 2 days
- **Description:** Write test suite covering all 15 AST node types, builtin function calls, texture sampling, SurfaceOutput construction, error cases (unsupported Python constructs -> clear error messages). Test that naga accepts all generated WGSL.
- **Acceptance:** 50+ tests covering positive and negative cases. All pass.
- **Dependencies:** T-MAT-1.3, T-MAT-1.4, T-MAT-1.6

---

## Phase 2: Shader Infrastructure (Weeks 5-8)

Gaps: S3-G3, S3-G4, S3-G10

### T-MAT-2.1: Variant const system
- **Gap:** S3-G3 (CRITICAL)
- **Effort:** 2 days
- **Description:** Design and implement const-boolean variant selection system. Define WGSL const declarations for domain, blend, and quality. Implement code in PBR template that gates optional features behind const bools: lighting loops, shadow sampling, advanced shading.
- **Acceptance:** Setting different const bools produces different naga IR (verified by IR comparison). naga dead-code elimination removes inactive branches.
- **Dependencies:** T-MAT-1.3

### T-MAT-2.2: Domain variants
- **Gap:** S3-G3 (CRITICAL)
- **Effort:** 2 days
- **Description:** Implement domain-specific variant logic for all 5 domains: surface (full PBR), deferred_decal (normal+color only), volume (single-scattering), post_process (fullscreen tonemap), ui (unlit vertex-color). Each domain gets its own const bool and gated code.
- **Acceptance:** All 5 domain variants compile to valid WGSL. Each produces observably different output structure.
- **Dependencies:** T-MAT-2.1

### T-MAT-2.3: Blend mode variants
- **Gap:** S3-G3 (CRITICAL)
- **Effort:** 1 day
- **Description:** Implement blend mode variant logic for all 5 blend modes: opaque, masked (alpha test with discard), translucent (alpha blend, transmissive lighting), additive, modulate. Each mode controls blend state, depth write, and lighting behavior.
- **Acceptance:** All 5 blend mode variants compile. Masked mode generates discard statements. Translucent mode skips depth write.
- **Dependencies:** T-MAT-2.1

### T-MAT-2.4: Quality tier variants
- **Gap:** S3-G3, S3-G9 (CRITICAL + HIGH)
- **Effort:** 2 days
- **Description:** Implement quality tier variant logic for low/medium/high. Low: 1 light, no shadows, no advanced shading. Medium: 4 lights, basic shadows, limited advanced. High: unlimited lights, PCSS shadows, full advanced shading.
- **Acceptance:** All 3 quality tiers compile. Low quality variant produces observably simpler naga IR.
- **Dependencies:** T-MAT-2.1

### T-MAT-2.5: Shader include system
- **Gap:** S3-G10 (HIGH)
- **Effort:** 2 days
- **Description:** Implement #include directive preprocessor for WGSL. Support include search paths (material-local, project-global). Recursive resolution with cycle detection and max depth limit. Record dependency edges in DepGraph during resolution.
- **Acceptance:** `#include "pbr/brdf.wgsl"` resolves correctly. Cyclic includes produce error. DepGraph records edges.
- **Dependencies:** T-MAT-1.3

### T-MAT-2.6: DepGraph implementation
- **Gap:** S3-G4 (CRITICAL)
- **Effort:** 2 days
- **Description:** Implement bidirectional dependency graph: `include_to_materials`, `material_to_includes`, `material_to_dependents` HashMaps. Implement `broadest_invalidation_set(path)` with BFS traversal. Implement edge recording on material compilation and include resolution. RwLock-guarded for concurrent access.
- **Acceptance:** Inserting 10 materials produces correct adjacency. BFS returns transitive closure. Lock contention < 1us under concurrent reads.
- **Dependencies:** T-MAT-2.5

### T-MAT-2.7: File watcher and hot-reload loop
- **Gap:** S3-G4 (CRITICAL)
- **Effort:** 3 days
- **Description:** Implement file watcher using cross-platform crate (notify). Watch material source directories. On file change: debounce for 500ms, query DepGraph for invalidation set, recompile each invalidated material, atomically swap in PipelineTable. Handle compilation errors gracefully (keep old pipeline, log error).
- **Acceptance:** Editing a material .py file triggers recompilation within 1 second. PipelineTable swap is atomic. Failed compilation preserves old pipeline.
- **Dependencies:** T-MAT-2.6

---

## Phase 3: PBR Core (Weeks 9-11)

Foundational BRDF -- no explicit gap ID

### T-MAT-3.1: WGSL PBR struct definitions
- **Effort:** 1 day
- **Description:** Define PBRInput, PBRParams, PBROutput WGSL structs. PBRInput: world_position, world_normal, world_view, uv, vertex_color, light_count, lights array. PBRParams: base_color, normal, roughness, metallic, specular, occlusion, emissive. PBROutput: color, depth. Implement in shared include file.
- **Acceptance:** Structs compile in naga. All fields have correct types.
- **Dependencies:** T-MAT-1.3

### T-MAT-3.2: Cook-Torrance BRDF functions
- **Effort:** 2 days
- **Description:** Implement GGX NDF (Trowbridge-Reitz), Smith-GGX GSF (height-correlated), Schlick Fresnel, and Cook-Torrance BRDF in WGSL. See PHASE_N_ARCH.md for exact function signatures. Test with reference values against known BRDF data.
- **Acceptance:** BRDF output matches reference within 1% for 20+ test inputs.
- **Dependencies:** T-MAT-3.1

### T-MAT-3.3: Light loop and shading
- **Effort:** 2 days
- **Description:** Implement directional/point/spot light types in WGSL. Implement light loop that accumulates BRDF output for N active lights. Implement shadow sample function (placeholder). Implement ambient occlusion term. Implement emissive term.
- **Acceptance:** 1-8 lights produce correct accumulated output. Shadow placeholder returns 1.0.
- **Dependencies:** T-MAT-3.2

### T-MAT-3.4: Rust pipeline integration
- **Effort:** 2 days
- **Description:** Implement ShaderCache (HashMap<u64, ShaderModule> with content-addressed key). Implement PipelineTable (HashMap<u32, CachedPipeline> with LRU eviction). Create wgpu pipeline from compiled WGSL. Wire PBRShader into ShaderCache -> PipelineTable -> frame graph. Render first PBR-shaded mesh.
- **Acceptance:** PBR-shaded triangle/mesh renders on screen. Pipeline cache hits verify identical output.
- **Dependencies:** T-MAT-3.3, T-MAT-2.7

### T-MAT-3.5: PBR validation suite
- **Effort:** 2 days
- **Description:** Write comprehensive PBR test suite. Test BRDF functions with known reference values. Test light accumulation. Test edge cases: roughness=0 (perfect mirror), roughness=1 (Lambertian), metallic=1 (no diffuse), metallic=0 (dielectric). Visual test: render sphere under known lighting, compare to reference.
- **Acceptance:** 30+ tests. All pass. DeltaE < 1.0 for reference renders.
- **Dependencies:** T-MAT-3.4

---

## Phase 4: Advanced Shading (Weeks 12-14)

Gaps: S3-G5, S3-G6, S3-G8

### T-MAT-4.1: Subsurface scattering implementation
- **Gap:** S3-G6 (MEDIUM)
- **Effort:** 3 days
- **Description:** Implement dual-pass screen-space SSS. Pass 1: Evaluate lighting, store irradiance in half-res buffer. Pass 2: Burley normalized diffusion separable blur (12-24 taps, horizontal then vertical, dual-buffer ping-pong). Importance-sampled kernel precomputation. Integration with quality tier const bool gating.
- **Acceptance:** SSS produces visible scattering on lit geometry. Kernel tap count matches quality tier. Blur is separable-correct.
- **Dependencies:** T-MAT-3.4

### T-MAT-4.2: Clear coat implementation
- **Gap:** S3-G6 (MEDIUM, part of advanced shading spec)
- **Effort:** 2 days
- **Description:** Implement dual-layer clear coat BRDF. Top layer: fixed IOR 1.5, independent roughness, Schlick Fresnel (F0=0.04). Bottom layer: standard Cook-Torrance. Layer combination via Fresnel-weighted blend. Integration with quality tier gating.
- **Acceptance:** Clear coat produces visible specular layer on top of base BRDF. Layer separation is Fresnel-weighted correctly.
- **Dependencies:** T-MAT-3.2

### T-MAT-4.3: Anisotropy implementation
- **Gap:** (S3 advanced spec, related to S3-G6 area)
- **Effort:** 2 days
- **Description:** Implement anisotropic GGX NDF with alpha_x/alpha_y derived from roughness + anisotropy_strength. Implement stretched tangent-space BRDF evaluation. Anisotropy direction parameter for tangent rotation. Integration with quality tier gating.
- **Acceptance:** Anisotropic BRDF produces directionally stretched highlights. Varying anisotropy_strength from 0 to 1 produces visible change.
- **Dependencies:** T-MAT-3.2

### T-MAT-4.4: Sheen implementation
- **Gap:** (S3 advanced spec)
- **Effort:** 1 day
- **Description:** Implement microfiber retro-reflection sheen lobe. Third BRDF lobe added to output. Separate NDF with low roughness. No Fresnel. sheen_color tint parameter. Integration with quality tier gating.
- **Acceptance:** Sheen adds visible retro-reflective tint. Disabling sheen (const bool) removes the lobe.
- **Dependencies:** T-MAT-3.2

### T-MAT-4.5: Transmission implementation
- **Gap:** S3-G8 (MEDIUM)
- **Effort:** 3 days
- **Description:** Implement thin-walled transmission model. Screen-space refraction: thickness map -> UV offset computation. Scene color sampling at offset UV. Beer's law absorption: exp(-absorption_coefficient * thickness). Combined output: transmitted * absorbed + reflected. Integration with quality tier gating.
- **Acceptance:** Glass-like transmission visible. Thicker areas show more absorption. Refraction UV offset follows screen-space direction.
- **Dependencies:** T-MAT-3.4

### T-MAT-4.6: Iridescence implementation
- **Gap:** S3-G5 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement thin-film interference iridescence. Film phase computation from thickness + IOR. Air-film and film-substrate Fresnel evaluation. Interference color combination. Modulate base color with interference term. iridescence_ior (1.3-2.0) and iridescence_thickness (100-1000nm) parameters.
- **Acceptance:** Iridescence produces rainbow-like color variation. Changing thickness shifts interference pattern.
- **Dependencies:** T-MAT-3.2

---

## Phase 5: Material System (Weeks 15-18)

Gaps: S3-G9, S3-G11, S3-G12, S3-G13, S3-G14, S3-G15, S3-G16

### T-MAT-5.1: Quality-driven variant compilation
- **Gap:** S3-G9 (HIGH)
- **Effort:** 2 days
- **Description:** Implement MaterialMeta-based triple compilation: each material compiled once per quality tier (low/medium/high) with different const bool sets. MaterialRegistry stores variant_key -> ShaderModule mapping. Runtime selection via `select_material_variant(material, quality)`.
- **Acceptance:** Each material has 3 compiled variants. Switching quality tier at runtime selects correct variant.
- **Dependencies:** T-MAT-2.4

### T-MAT-5.2: Material inheritance model
- **Gap:** S3-G11 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement Python class inheritance for MaterialMeta classes. Child inherits texture slot declarations. Child can override surface() method. super() calls in AST produce parent.surface() WGSL inlining. MRO resolution for combined WGSL output.
- **Acceptance:** Child class inheriting from parent produces WGSL that includes parent surface logic + child overrides. super() calls work correctly.
- **Dependencies:** T-MAT-1.2

### T-MAT-5.3: Decal domain implementation
- **Gap:** S3-G12 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement deferred decal material domain. Screen-space UV computation from decal projection matrix. G-buffer normal + base_color modification (no lighting evaluation). Blend modes: opaque (replace) and translucent (alpha blend). Decal-specific PBRInput fields.
- **Acceptance:** Decal material renders in decal pass and modifies G-buffer. No lighting evaluation occurs.
- **Dependencies:** T-MAT-2.2, T-MAT-3.4

### T-MAT-5.4: Volume domain implementation
- **Gap:** S3-G13 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement volume material domain. Camera ray-volume AABB intersection. Single-scattering approximation: in-scattering + out-scattering integral. density, absorption_color, scattering_color, phase_anisotropy (g) parameters. No BRDF evaluation.
- **Acceptance:** Volume material renders visible fog/smoke effect. Increasing density produces more opacity.
- **Dependencies:** T-MAT-2.2, T-MAT-3.4

### T-MAT-5.5: Material animation system
- **Gap:** S3-G14 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement ctx.time() built-in mapped to WGSL uniform buffer updated each frame. Support for time-driven parameter modulation in DSL (sin/cos/mix with time input). Ensure no CPU-side recompilation needed for animation parameter changes.
- **Acceptance:** ctx.time() produces incrementing value in WGSL. sin(time()) modulates color correctly. No recompilation on time change.
- **Dependencies:** T-MAT-1.6, T-MAT-3.4

### T-MAT-5.6: Material LOD system
- **Gap:** S3-G15 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement distance-based material LOD selection. lod_distances threshold array, lod_materials per-level array. LOD 0 = full quality, LOD 3 = unlit fallback. Blend option for cross-fade between LOD levels. Integration with runtime quality tier selection.
- **Acceptance:** Material switches to lower LOD at distance thresholds. Cross-fade transition (when enabled) is smooth.
- **Dependencies:** T-MAT-5.1

### T-MAT-5.7: Bindless texture arrays
- **Gap:** S3-G16 (HIGH)
- **Effort:** 3 days
- **Description:** Implement bindless resource binding model. Replace per-texture bindings with binding_array<texture_2d<f32>> and binding_array<sampler>. Add texture_index uniform array for per-material texture selection. Verify WebGPU maxSampledTexturesPerShaderStage limits. Fallback to bindful mode on devices that don't support bindless.
- **Acceptance:** Bindless mode uses single bind group for all textures. Material change updates only index array. Fallback mode works on restricted devices.
- **Dependencies:** T-MAT-3.4

### T-MAT-5.8: UI material domain
- **Gap:** (variant coverage)
- **Effort:** 1 day
- **Description:** Implement UI material domain. Unlit shading: vertex-color only, no PBR, no lighting. Fullscreen UV coordinates. Screen-space position. Simple base_color * vertex_color output.
- **Acceptance:** UI material renders correct screen-space output. No lighting evaluation in generated WGSL.
- **Dependencies:** T-MAT-2.2

---

## Phase 6: Content Store Foundation (Weeks 19-21)

Gaps: S16-G1, S16-G18, S16-G19, S16-G25

### T-MAT-6.1: ContentHash and SHA-256 implementation
- **Gap:** S16-G1 (HIGH)
- **Effort:** 1 day
- **Description:** Implement ContentHash newtype wrapping [u8; 32]. Implement SHA-256 hashing for content data. Implement Display/Debug in hex format. Implement FromStr for hex parsing. Implement PartialEq/Eq/Hash for use as HashMap key.
- **Acceptance:** ContentHash round-trips through hex string. Hashing produces correct SHA-256 output.
- **Dependencies:** None

### T-MAT-6.2: FileBackend content store
- **Gap:** S16-G18, S16-G19 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement FileBackend: put/get/has/tree_put/tree_get. Directory layout: root/hash[:2]/hash[2:]/data. Streaming write for data > STREAMING_THRESHOLD. Memory-mapped read for data > MMAP_THRESHOLD. Concurrent read safety (no write conflicts since put is content-addressed).
- **Acceptance:** put/get round-trips for small and large data. Directory structure matches git-style layout. Concurrent reads succeed.
- **Dependencies:** T-MAT-6.1

### T-MAT-6.3: ContentTree with structural sharing
- **Gap:** S16-G18 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement ContentTree with __tree_type__, __children__, __data__, __content_hash__ sentinel fields. Serialization/deserialization (to bytes for ContentStore storage). Structural sharing: identical subtrees produce identical ContentHash. Tree diff via hash comparison.
- **Acceptance:** Two trees with identical children produce same hash. Tree round-trips through content store.
- **Dependencies:** T-MAT-6.2

### T-MAT-6.4: BLAKE3 implementation (optional upgrade)
- **Gap:** S16-G1 (HIGH)
- **Effort:** 1 day
- **Description:** Implement BLAKE3 hashing as optional ContentHash backend. Feature-gated compilation: default SHA-256, optional BLAKE3. Hash format tag in ContentHash (byte 0 distinguishes algorithm). Runtime algorithm selection.
- **Acceptance:** BLAKE3 produces correct output. Switching algorithm at compile time works.
- **Dependencies:** T-MAT-6.1

### T-MAT-6.5: Pipeline cache sharding
- **Gap:** S16-G25 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement ShardedPipelineCache with `shard_index = pipeline_hash % shard_count`. Per-shard RwLock for independent concurrent access. NUMA-aware shard assignment. Graceful fallback to single shard. Tune shard count based on CPU core count.
- **Acceptance:** 4-shard cache supports 4 concurrent lookups without contention. Fallback to single shard works.
- **Dependencies:** T-MAT-3.4

---

## Phase 7: Content Store Advanced (Weeks 22-24)

Gaps: S16-G2, S16-G20, S16-G21, S16-G22, S16-G23

### T-MAT-7.1: Streaming API for large content
- **Gap:** S16-G2 (HIGH)
- **Effort:** 2 days
- **Description:** Implement put_stream/get_stream on ContentStore trait. Chunked content storage: large files split into fixed-size (256KB) chunks, each independently content-addressed. Chunk list stored in ContentTree node. Streaming reader/writer interfaces. Verify with 1GB+ test data.
- **Acceptance:** 1GB file stored and retrieved via streaming API. Memory usage < 10MB during streaming.
- **Dependencies:** T-MAT-6.3

### T-MAT-7.2: ContentDiffer implementation
- **Gap:** S16-G20 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement ContentDiffer trait with diff/apply methods. BinaryDiffer specialization using bsdiff. TreeDiffer specialization with recursive walk. Delta: enum with Full, BinaryPatch, TreeDiff, ParameterPatch variants. test with known diffs.
- **Acceptance:** Binary diff produces patch < 50% of full size for similar inputs. Tree diff correctly identifies changed children.
- **Dependencies:** T-MAT-6.3

### T-MAT-7.3: DeltaSync protocol
- **Gap:** S16-G21 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement DeltaSync protocol: client-server sync with checkpoint-based incremental transfer. Protocol version negotiation. Delta compression (zstd). Batch deltas with ordering. Remove list for deleted assets. Test with simulated remote store.
- **Acceptance:** Sync 1000 assets in < 5 seconds. Incremental sync transfers only changed content. Version negotiation works.
- **Dependencies:** T-MAT-7.2

### T-MAT-7.4: Tree store garbage collection
- **Gap:** S16-G22 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement mark-and-sweep GC for ContentTree store. Mark phase: BFS from root set (active materials, assets, pipeline cache). Sweep phase: delete unmarked trees, unreferenced FileBackend objects. Background thread with per-frame time budget. Reference counting for immediate cleanup.
- **Acceptance:** Orphaned trees are collected. Active trees are preserved. GC runs within 2ms frame budget.
- **Dependencies:** T-MAT-6.3

### T-MAT-7.5: Provenance chain pruning
- **Gap:** S16-G23 (MEDIUM)
- **Effort:** 1 day
- **Description:** Implement provenance chain PruningStrategy: keep-last-N (default 10), max-age (default 30 days), always-keep-first (origin), always-keep-last (current). Automatic pruning on new provenance entry addition.
- **Acceptance:** Chain exceeding N entries is trimmed. Origin and current entries are always preserved.
- **Dependencies:** T-MAT-6.3

---

## Phase 8: Mesh Pipeline (Weeks 25-27)

Gaps: S16-G5, S16-G8, S16-G15

### T-MAT-8.1: glTF mesh loader
- **Gap:** (foundational S16 asset loading)
- **Effort:** 2 days
- **Description:** Implement glTF 2.0 mesh loader. Parse glTF JSON + binary buffers. Extract vertex attributes (position, normal, tangent, UV, color). Extract index buffers. Support interleaved and split vertex formats. Handle 8/16/32-bit index formats. Support glTF extensions (KHR_draco_mesh_compression optional, KHR_materials_* passthrough).
- **Acceptance:** Loads standard glTF 2.0 test models. Vertex and index data matches glTF spec.
- **Dependencies:** T-MAT-6.2

### T-MAT-8.2: Meshlet generation
- **Gap:** (foundational S16)
- **Effort:** 3 days
- **Description:** Implement meshlet partitioner: 64 max unique vertices, 124 max triangles per meshlet. Morton order spatial sorting for locality. Meshlet bounds computation (bounding sphere). Vertex buffer conversion to meshlet-local indices. Output: Vec<Meshlet> with vertex and index data.
- **Acceptance:** Large mesh generates multiple meshlets. No meshlet exceeds 64/124 limits. Meshlet bounds are tight.
- **Dependencies:** T-MAT-8.1

### T-MAT-8.3: BLAS construction
- **Gap:** (foundational S16)
- **Effort:** 2 days
- **Description:** Implement BLAS (Bottom-Level Acceleration Structure) for ray tracing. Use ALLOW_COMPACTION flag for optimal memory. Use ALLOW_UPDATE flag for animated meshes. AS compaction post-build. Integration with wgpu BLAS API.
- **Acceptance:** BLAS builds for test mesh. Compaction reduces memory. Update flag allows in-place update.
- **Dependencies:** T-MAT-8.2

### T-MAT-8.4: LOD generation and blending
- **Gap:** S16-G5 (HIGH), S16-G15 (MEDIUM)
- **Effort:** 3 days
- **Description:** Implement discrete LOD generation: N LOD levels via mesh simplification. LOD blending: alpha-crossfade, dither-pattern transition. LOD bias per-viewport for editor/VR tuning. Integration with LOD selection in render pass.
- **Acceptance:** 3 LOD levels generated. Cross-fade transitions are smooth. Viewport bias shifts LOD selection.
- **Dependencies:** T-MAT-8.2

### T-MAT-8.5: Budget-aware LOD selection
- **Gap:** S16-G8 (HIGH)
- **Effort:** 2 days
- **Description:** Implement BudgetTracker with per-category budgets (mesh 512MB, texture 1GB, shader 256MB, global 2GB). Priority-sorted LOD assignment: compute priority for each visible mesh, sort descending, assign LOD to fit budget. Current usage snapshot for budget accounting.
- **Acceptance:** Scene with over-budget meshes automatically reduces LOD for low-priority meshes. Budget enforcement works.
- **Dependencies:** T-MAT-8.4

---

## Phase 9: Texture Pipeline (Weeks 28-30)

Gaps: S16-G11, S16-G17, S16-G24

### T-MAT-9.1: Texture importer plugin system
- **Gap:** S16-G24 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement FormatImporter trait with format(), import(), priority() methods. ImporterRegistry with register/resolve. Implement importers: PNG, BMP, TGA, JPEG, WebP via Rust image crate. EXR, TIFF, PSD, DDS, KTX/KTX2 via format-specific crates. Plugin discovery via feature flags.
- **Acceptance:** PNG/JPEG/EXR importers produce correct pixel data. Importer resolution by format string works.
- **Dependencies:** T-MAT-6.2

### T-MAT-9.2: Format selection and cooking pipeline
- **Gap:** (foundational S16)
- **Effort:** 2 days
- **Description:** Implement texture cooking: source -> format selection -> mip generation -> GPU upload. Format selection: base_color -> SRGB/BC7, normal_map -> UNORM/BC5, roughness/metallic/occlusion -> BC4, emissive -> BC6H, data -> R32F. Mip generation via Lanczos filtering. GPU upload via dedicated transfer queue.
- **Acceptance:** Texture source->cooked->GPU round-trip produces correct rendering. Format selection matches heuristics.
- **Dependencies:** T-MAT-9.1

### T-MAT-9.3: Virtual texturing system
- **Gap:** S16-G17 (MEDIUM)
- **Effort:** 3 days
- **Description:** Implement virtual texture: 128x128 page size, physical texture atlas, page table in GPU buffer, feedback buffer (GPU writes page requests), CPU-side page cache (LRU). Page table lookup in shader: page_coord -> page_entry -> physical_uv -> sample. Feedback analysis: schedule page loads, update page table.
- **Acceptance:** Virtual texture renders correctly. Feedback buffer identifies visible pages. Page cache LRU eviction works.
- **Dependencies:** T-MAT-9.2

### T-MAT-9.4: Cubemap, texture array, and cubemap array support
- **Gap:** (foundational S16)
- **Effort:** 1 day
- **Description:** Implement cubemap import from cross/vertical/horizontal layouts. Texture array support for array textures. Cubemap array for dynamically-generated cubemaps (reflection probes). Generate all mip levels for all formats.
- **Acceptance:** Cubemap renders correctly. Texture arrays are mipmapped. Cubemap arrays sample correctly.
- **Dependencies:** T-MAT-9.2

### T-MAT-9.5: High-priority format importer implementation
- **Gap:** S16-G11 (HIGH)
- **Effort:** 3 days
- **Description:** Implement format importers for: USD/USDZ (via usd-skel or alternative Rust crate), KTX2/Basis Universal (via ktx2 crate + transcoder), FBX (via ufbx Rust bindings). Each importer produces standardized intermediate representation for cooking pipeline.
- **Acceptance:** USD, KTX2, FBX test files import correctly. Imported data matches source.
- **Dependencies:** T-MAT-9.1

---

## Phase 10: Asset Pipeline Integration (Weeks 31-34)

Gaps: S16-G3, S16-G4, S16-G6, S16-G7, S16-G9, S16-G10, S16-G12, S16-G13, S16-G14, S16-G16

### T-MAT-10.1: Predictive pre-loading system
- **Gap:** S16-G3 (HIGH)
- **Effort:** 2 days
- **Description:** Implement PreloadPredictor with heuristic model: visibility, velocity, distance, LOD bias weighted scoring. Configurable weights. Predict next visible assets based on camera movement direction. Submit pre-load requests to asset streaming queue. Integration with camera system.
- **Acceptance:** Pre-loading reduces visible pop-in by > 50%. Pre-load requests do not starve active loading.
- **Dependencies:** T-MAT-8.5

### T-MAT-10.2: Priority queue with starvation prevention
- **Gap:** S16-G7 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement PriorityQueue with 4 tiers, per-tier locks, starvation counter, and batch dequeue. Starvation promotion: low-prio items waiting > N frames promoted to normal priority. Configurable tier weights. Performance: < 1us per enqueue/dequeue.
- **Acceptance:** Starving low-prio item is promoted after N frames. Batch dequeue returns up to N items.
- **Dependencies:** T-MAT-8.5

### T-MAT-10.3: Cache TTL and database-backed cache
- **Gap:** S16-G4, S16-G6 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement per-entry TTL expiry in cache. SQLite-backed metadata store: cache_entries table with hash, size, created_at, last_access, access_count, ttl, flags. Binary data remains in FileBackend. Eviction: expired first, then least-recently-used.
- **Acceptance:** Expired entries are evicted on next access. SQLite query for eviction candidates takes < 1ms.
- **Dependencies:** T-MAT-6.2

### T-MAT-10.4: Remote cache distribution
- **Gap:** S16-G12 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement remote content cache accessible via HTTP. Client: check remote cache on local miss, fetch + store locally. Server: HTTP API for put/get/has. DeltaSync integration: remote cache serves as DeltaSync source. Authentication via token. Compression (zstd) for transfer.
- **Acceptance:** Client fetches from remote cache on local miss. Subsequent access uses local cache.
- **Dependencies:** T-MAT-7.3

### T-MAT-10.5: Incremental build system
- **Gap:** S16-G10 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement dependency-tracked incremental builds. Each asset records source file hashes + settings hash. Build graph: asset -> dependency list. Change detection via hash comparison. Cache: ContentStore-based build artifact storage tagged with build configuration hash.
- **Acceptance:** Rebuilding with unchanged sources completes in < 100ms. Single source change triggers rebuild of only affected assets.
- **Dependencies:** T-MAT-6.2

### T-MAT-10.6: Shader edit-and-continue with hot-reload
- **Gap:** S16-G9, S16-G13 (MEDIUM)
- **Effort:** 2 days
- **Description:** Integrate S3 hot-reload with asset pipeline. Edit-and-continue: developer edits .wgsl or DSL .py -> file watcher -> DepGraph invalidation -> recompile -> PipelineTable atomic swap -> frame graph rebuild. Graceful error handling: failed compilation logs error, keeps old pipeline.
- **Acceptance:** Editing WGSL triggers hot-reload in < 1s. Failed compilation preserves working state. No process restart needed.
- **Dependencies:** T-MAT-2.7, T-MAT-10.5

### T-MAT-10.7: Material hot-reload (parameter only)
- **Gap:** S16-G14 (MEDIUM)
- **Effort:** 2 days
- **Description:** Implement parameter-only material hot-reload. Bridge Data channel propagates parameter changes. Uniform buffer update on next frame. No shader recompilation for parameter changes. Shader changes via edit-and-continue (T-MAT-10.6). DepGraph update on material struct change.
- **Acceptance:** Changing material parameter updates rendering within 1 frame. No pipeline rebuild on parameter change.
- **Dependencies:** T-MAT-10.6

### T-MAT-10.8: Streaming heuristics tuning
- **Gap:** S16-G16 (MEDIUM)
- **Effort:** 1 day
- **Description:** Implement feedback-based streaming heuristics tuning. Monitor: page miss rate, load latency, budget pressure, LOD switches per second. Auto-tune: pre-load distance, urgency threshold, tier weights. Expose tuning parameters for developer override. Logging for analysis.
- **Acceptance:** Heuristics adapt to scene complexity. Override parameters work. Miss rate decreases over session time.
- **Dependencies:** T-MAT-10.1, T-MAT-10.2

---

## Phase 11: Hardening (Weeks 35-37)

No new gaps -- system integration, testing, and validation

### T-MAT-11.1: End-to-end rendering test suite
- **Effort:** 3 days
- **Description:** Write end-to-end tests: DSL compile -> WGSL -> naga -> pipeline -> render output. Test all 75 variant combinations compile. Test PBR output matches reference. Test all advanced shading models. Test hot-reload cycle. Test asset pipeline import->cook->load->render.
- **Acceptance:** 100+ end-to-end tests. All pass. 75 variants all produce valid WGSL.
- **Dependencies:** All prior phases

### T-MAT-11.2: Visual regression testing
- **Effort:** 2 days
- **Description:** Implement screenshot comparison testing. Render reference scenes, capture output, compare to stored reference images. DeltaE perceptual difference metric. Per-pixel error visualization. Automated CI integration.
- **Acceptance:** Identical renders produce < 0.5% pixel difference. Deliberate regression produces > 5% difference.
- **Dependencies:** T-MAT-11.1

### T-MAT-11.3: Performance benchmarking
- **Effort:** 2 days
- **Description:** Benchmark: DSL compile time (target < 10ms per material), naga compile time, pipeline cache hit rate (target > 90%), hot-reload latency (target < 100ms), content store lookup (target < 1ms), texture cooking (target < 50ms per 2K texture). Benchmark automation and regression detection.
- **Acceptance:** All benchmarks meet targets. CI catches regressions > 5%.
- **Dependencies:** All prior phases

### T-MAT-11.4: Memory and leak audit
- **Effort:** 2 days
- **Description:** Memory audit: DepGraph adjacency growth, PipelineTable LRU effectiveness, ShaderCache unbounded growth, ContentStore orphan detection, hot-reload accumulation. Fix leaks. Add memory pressure tests (10,000 materials, 100,000 assets).
- **Acceptance:** Memory stabilizes under load. No unbounded growth. DepGraph, PipelineTable, ShaderCache within budget.
- **Dependencies:** T-MAT-11.3

### T-MAT-11.5: Bridge protocol stress testing
- **Effort:** 2 days
- **Description:** Stress test Bridge Data channel: 10,000+ component writes per frame. Verify < 100ns per-field latency target. Test Type channel with 1000+ material registrations. Test Command channel with concurrent resize/screenshot/shutdown requests.
- **Acceptance:** Data channel < 100ns per field. Type channel handles 1000 registrations. Command channel handles concurrent requests.
- **Dependencies:** T-MAT-11.1

### T-MAT-11.6: Cross-platform and security validation
- **Effort:** 3 days
- **Description:** Test on Linux (Vulkan), macOS (Metal via MoltenVK), Windows (Vulkan). SOTA parity verification for all SOTA_COMPARISON.md claims. Security audit: AST injection in DSL, include path traversal, shader compilation sandboxing, content store path validation.
- **Acceptance:** All platforms pass rendering tests. SOTA claims verified. Security audit passes with no critical findings.
- **Dependencies:** T-MAT-11.1

---

## Task Summary

| Phase | Name | Tasks | Gaps |
|-------|------|-------|------|
| 1 | DSL Foundation | 7 | S3-G1, S3-G2, S3-G7, S3-G17 |
| 2 | Shader Infrastructure | 7 | S3-G3, S3-G4, S3-G10 |
| 3 | PBR Core | 5 | (foundational) |
| 4 | Advanced Shading | 6 | S3-G5, S3-G6, S3-G8 |
| 5 | Material System | 8 | S3-G9, S3-G11, S3-G12, S3-G13, S3-G14, S3-G15, S3-G16 |
| 6 | Content Store Foundation | 5 | S16-G1, S16-G18, S16-G19, S16-G25 |
| 7 | Content Store Advanced | 5 | S16-G2, S16-G20, S16-G21, S16-G22, S16-G23 |
| 8 | Mesh Pipeline | 5 | S16-G5, S16-G8, S16-G15 |
| 9 | Texture Pipeline | 5 | S16-G11, S16-G17, S16-G24 |
| 10 | Asset Pipeline Integration | 8 | S16-G3, G4, G6, G7, G9, G10, G12, G13, G14, G16 |
| 11 | Hardening | 6 | (all) |
| **Total** | | **67 tasks** | **43 gaps** |
