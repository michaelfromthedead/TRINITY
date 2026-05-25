# GAPSET_11_DEMOSCENE -- Task List with TASK_IDs

> **TASK_ID Format**: T-DEMO-{PHASE}.{N}
> **Total Tasks**: 46
> **All Severities**: LOW (S13 is standalone, does not block any subsystem)

---

## Phase 1: SDF Primitive Library and Combinators

### WGSL Primitive Functions

- [ ] **T-DEMO-1.1**: Implement `sdf_sphere(p, r)` in WGSL as a pure function. Acceptance: returns correct signed distance for center, surface, exterior, diagonal, negative radius.
- [ ] **T-DEMO-1.2**: Implement `sdf_box(p, b)` in WGSL. Acceptance: correct distance for inside corner, outside corner, edge center, face center.
- [ ] **T-DEMO-1.3**: Implement `sdf_torus(p, r)` in WGSL. Acceptance: correct distance inside hole, on major radius, on minor radius.
- [ ] **T-DEMO-1.4**: Implement `sdf_cylinder(p, h)` in WGSL. Acceptance: correct distance inside, on surface, above cap, below cap.
- [ ] **T-DEMO-1.5**: Implement `sdf_cone(p, c)` in WGSL. Acceptance: signed distance to conical surface matches reference.
- [ ] **T-DEMO-1.6**: Implement `sdf_plane(p, n)` in WGSL. Acceptance: correct distance on plane, above, below.
- [ ] **T-DEMO-1.7**: Implement `sdf_capsule(p, a, b, r)` in WGSL. Acceptance: correct distance at endpoints, midpoint, radius test.
- [ ] **T-DEMO-1.8**: Implement `sdf_ellipsoid(p, r)` in WGSL. Acceptance: correct signed distance via normalization.
- [ ] **T-DEMO-1.9**: Implement `sdf_box_frame(p, b, e)` in WGSL. Acceptance: hollow box with correct thickness.
- [ ] **T-DEMO-1.10**: Implement `sdf_rounded_box(p, b, r)` in WGSL. Acceptance: box with rounded corners at specified radius.
- [ ] **T-DEMO-1.11**: Implement `sdf_octahedron(p, s)` in WGSL. Acceptance: correct distance to octahedral surface.
- [ ] **T-DEMO-1.12**: Implement `sdf_pyramid(p, h)` in WGSL. Acceptance: correct signed distance to square pyramid.

**Dependencies**: None. These are independent pure functions.
**Estimated Effort**: 3-5 days.

### WGSL Combinator Functions

- [ ] **T-DEMO-1.13**: Implement `min2(a, b)` for vec2 comparison (compare by x, carry y of winner). Acceptance: correctly selects winning material ID.
- [ ] **T-DEMO-1.14**: Implement `max2(a, b)` for vec2 comparison. Acceptance: intersection with correct material propagation.
- [ ] **T-DEMO-1.15**: Implement union combinator (vec2 variant). Acceptance: surface exists in either operand.
- [ ] **T-DEMO-1.16**: Implement intersection combinator (vec2 variant). Acceptance: surface exists in both operands.
- [ ] **T-DEMO-1.17**: Implement subtraction combinator (vec2 variant). Acceptance: surface of a with b carved out, correct material from a.
- [ ] **T-DEMO-1.18**: Implement smooth union `smin(a, b, k)` with polynomial blend. Acceptance: C1 continuous blend at junction.
- [ ] **T-DEMO-1.19**: Implement smooth intersection `smax(a, b, k)` with polynomial blend. Acceptance: C1 continuous blend at junction.
- [ ] **T-DEMO-1.20**: Implement smooth subtraction `smax(a, -b, k)` with polynomial blend. Acceptance: C1 continuous blend at junction.
- [ ] **T-DEMO-1.21**: Implement displacement combinator `sdf_displaced(p, base_sdf, amplitude, frequency)`. Acceptance: surface perturbed by FBM noise.

**Dependencies**: T-DEMO-1.1 through T-DEMO-1.12 (primitives).
**Estimated Effort**: 2-3 days.

### WGSL Domain Operations

- [ ] **T-DEMO-1.22**: Implement domain repetition via `mod(p, cell_size)`. Acceptance: infinite tiling with correct period.
- [ ] **T-DEMO-1.23**: Implement domain mirroring via `abs(p)`. Acceptance: reflection symmetry across specified plane.
- [ ] **T-DEMO-1.24**: Implement kaleidoscopic fold (KIFS). Acceptance: 6-iteration symmetric folding with scale compensation.
- [ ] **T-DEMO-1.25**: Implement twist operation (rotation proportional to height). Acceptance: space twisted along specified axis.
- [ ] **T-DEMO-1.26**: Implement bend operation (curvature of coordinate axes). Acceptance: shapes bent along specified radius.
- [ ] **T-DEMO-1.27**: Implement stretch operation (anisotropic scaling). Acceptance: space scaled along specified axis.

**Dependencies**: T-DEMO-1.1 through T-DEMO-1.12 (primitives for testing).
**Estimated Effort**: 2-3 days.

### Noise Functions (WGSL)

- [ ] **T-DEMO-1.28**: Implement hash functions for pseudo-random number generation. Acceptance: uniform distribution, no visible patterns.
- [ ] **T-DEMO-1.29**: Implement value noise (1D, 2D, 3D). Acceptance: smooth interpolation between hash values.
- [ ] **T-DEMO-1.30**: Implement Perlin noise (3D). Acceptance: gradient-based noise with zero mean.
- [ ] **T-DEMO-1.31**: Implement FBM (fractal Brownian motion) with configurable octaves, lacunarity, gain. Acceptance: multi-octave noise with correct spectral composition.
- [ ] **T-DEMO-1.32**: Implement ridged noise (1.0 - abs(FBM)) for terrain. Acceptance: sharp valleys, smooth ridges.
- [ ] **T-DEMO-1.33**: Implement domain warping for noise. Acceptance: FBM-warped FBM with increased visual complexity.

**Dependencies**: None (independent pure functions).
**Estimated Effort**: 2-4 days.

---

## Phase 2: Python SDF DSL Compiler

- [ ] **T-DEMO-2.1**: Design and implement the AST Builder -- recursively walk Python DSL objects and build expression tree. Acceptance: tree faithfully represents the scene graph.
- [ ] **T-DEMO-2.2**: Implement Trinity Pattern integration for DSL nodes (metaclasses, Mirror, Tracker). Acceptance: DSL nodes are valid Trinity Objects with introspection and dirty tracking.
- [ ] **T-DEMO-2.3**: Implement WGSL code generator for primitive nodes (Sphere, Box, Torus, etc.). Acceptance: generates correct `sdf_*` function calls.
- [x] **T-DEMO-2.4**: Implement WGSL code generator for combinator nodes (Union, Intersection, Subtraction, Smooth*). Acceptance: generates correct `min2`/`max2`/`smin`/`smax` with material ID propagation.
- [ ] **T-DEMO-2.5**: Implement WGSL code generator for domain operations (Repeat, Mirror, Fold, Twist, Bend, Stretch). Acceptance: generates correct domain transformation code.
- [ ] **T-DEMO-2.6**: Implement WGSL code generator for Material nodes. Acceptance: generates correct Material struct and scene_material function.
- [ ] **T-DEMO-2.7**: Implement WGSL code generator for Scene (camera, lights, render settings). Acceptance: generates complete compute shader with main(), scene_sdf(), scene_material().
- [ ] **T-DEMO-2.8**: Implement optimization pass: constant folding. Acceptance: pre-computed values are inlined, not computed at runtime.
- [ ] **T-DEMO-2.9**: Implement optimization pass: dead code elimination. Acceptance: unreachable branches are removed from the WGSL output.
- [ ] **T-DEMO-2.10**: Implement optimization pass: common sub-expression elimination. Acceptance: repeated computations are hoisted and reused.
- [ ] **T-DEMO-2.11**: Implement optimization pass: domain repetition flattening. Acceptance: nested repeats are flattened into iterative form where possible.
- [ ] **T-DEMO-2.12**: Implement optimization pass: material merging. Acceptance: adjacent same-material surfaces are merged.
- [ ] **T-DEMO-2.13**: Implement cached compilation with Tracker dirty invalidation. Acceptance: recompiling same scene returns cached WGSL; changing a parameter invalidates the cache.
- [ ] **T-DEMO-2.14**: Implement error reporting for invalid scenes (infinite recursion, impossible SDF, type errors). Acceptance: clear, actionable error messages.

**Dependencies**: Phase 1 (primitives and combinators are compilation targets).
**Estimated Effort**: 3-4 weeks.

---

## Phase 3: Ray Marching Compute Pipeline

- [ ] **T-DEMO-3.1**: Implement camera ray generation (pinhole model). Acceptance: rays are correctly generated from camera position, target, FOV, aspect ratio.
- [ ] **T-DEMO-3.2**: Implement the ray marching loop with sphere tracing. Acceptance: ray marches correctly, terminates at surface or max distance.
- [ ] **T-DEMO-3.3**: Implement perceptual termination criterion (epsilon scaled by ray length). Acceptance: distant objects terminate earlier, visual quality is acceptable.
- [ ] **T-DEMO-3.4**: Implement normal estimation via central differences (6-point stencil). Acceptance: normals are unit-length and smooth for known geometry.
- [ ] **T-DEMO-3.5**: Implement SDF ambient occlusion (Quilez's method, 5 evaluations). Acceptance: crevices are darkened, flat surfaces are not. Perceptually correct.
- [ ] **T-DEMO-3.6**: Implement soft SDF shadows (Quilez's penumbra, 32 steps). Acceptance: shadows with contact hardening. Penumbra width controllable via k parameter.
- [ ] **T-DEMO-3.7**: Implement diffuse lighting with multiple lights. Acceptance: correct Lambertian diffuse with shadow term.
- [ ] **T-DEMO-3.8**: Implement specular lighting (Blinn-Phong or GGX). Acceptance: specular highlights from shared S3 BRDF library.
- [ ] **T-DEMO-3.9**: Implement the full-screen compute shader dispatch (`@compute @workgroup_size(8, 8, 1)`). Acceptance: single dispatch covers entire viewport.
- [ ] **T-DEMO-3.10**: Implement sky color function for miss rays. Acceptance: gradient sky or solid color.
- [ ] **T-DEMO-3.11**: Implement tone mapping for output. Acceptance: HDR colors are mapped to display range.
- [ ] **T-DEMO-3.12**: Implement depth of field (optional lens jitter). Acceptance: out-of-focus blur at configurable aperture and focal distance.
- [ ] **T-DEMO-3.13**: Implement temporal anti-aliasing via sub-pixel jitter accumulation. Acceptance: converging to smooth image over N frames.

**Dependencies**: Phase 1 (primitives in WGSL), S14 (RHI for dispatch), S3 (BRDF for lighting).
**Estimated Effort**: 3-4 weeks.

---

## Phase 4: Procedural Worlds and Texture-Free Materials

- [ ] **T-DEMO-4.1**: Implement heightmap-based terrain SDF with FBM noise. Acceptance: continuous terrain surface from noise function.
- [ ] **T-DEMO-4.2**: Implement ridged noise terrain with sharp valleys and smooth ridges. Acceptance: realistic mountain terrain.
- [ ] **T-DEMO-4.3**: Implement domain-warped terrain for increased variety. Acceptance: warped noise produces non-repeating landscapes.
- [ ] **T-DEMO-4.4**: Implement 3D terrain SDF with overhangs and caves. Acceptance: FBM displacement creates caves, arches, overhangs.
- [ ] **T-DEMO-4.5**: Implement tree SDF (trunk + canopy spheres). Acceptance: tree shape with branching structure.
- [ ] **T-DEMO-4.6**: Implement infinite forest via domain repetition with pseudo-random variation. Acceptance: varied tree instances at each cell.
- [ ] **T-DEMO-4.7**: Implement building SDF (box structure with window carvings and roof). Acceptance: recognizable building from combined primitives.
- [ ] **T-DEMO-4.8**: Implement city block with domain repetition and per-block random variation. Acceptance: varied cityscape from repeated pattern.
- [ ] **T-DEMO-4.9**: Implement planet SDF (spherical terrain). Acceptance: spherical world with noise-based topography.
- [ ] **T-DEMO-4.10**: Implement Mandelbulb SDF. Acceptance: 3D fractal with correct distance estimation.
- [ ] **T-DEMO-4.11**: Implement KIFS (kaleidoscopic iterated function system) SDF. Acceptance: fractal geometry with scale-compensated distance.
- [ ] **T-DEMO-4.12**: Implement bump mapping from noise gradients (4 FBM evaluations per pixel). Acceptance: surface normals perturbed by noise field.
- [ ] **T-DEMO-4.13**: Implement surface curvature detection via Laplacian of noise. Acceptance: edges, creases, and ridges detected correctly.
- [ ] **T-DEMO-4.14**: Implement height-based terrain color palettes (water, sand, grass, rock, snow). Acceptance: smooth gradient transitions between zones.
- [ ] **T-DEMO-4.15**: Implement procedural palette patterns (stripes, checkerboard, wood grain, marble, rust). Acceptance: recognizable patterns from mathematical functions.
- [ ] **T-DEMO-4.16**: Implement 256-entry palette LUT for artistic color control. Acceptance: 1 KB texture lookup, optional per-material.

**Dependencies**: Phase 3 (ray marching pipeline for rendering).
**Estimated Effort**: 3-4 weeks.

---

## Phase 5: 4K/64K Size-Constrained Mode

- [ ] **T-DEMO-5.1**: Create minimal wgpu-rs standalone bootstrap (device, swapchain, queue). Acceptance: creates GPU device and swapchain in < 100 lines.
- [ ] **T-DEMO-5.2**: Create minimal window/presentation layer. Acceptance: window appears, swapchain presents frames.
- [ ] **T-DEMO-5.3**: Embed WGSL shader as string literal in Rust binary. Acceptance: shader compiles and runs from embedded source.
- [ ] **T-DEMO-5.4**: Implement render loop (update, dispatch, present, poll). Acceptance: continuous rendering at display refresh rate.
- [ ] **T-DEMO-5.5**: Build-time DSL compilation pipeline (Python DSL -> WGSL -> embedded in Rust). Acceptance: build system invokes DSL compiler, embeds output in executable.
- [ ] **T-DEMO-5.6**: Optimize binary size (strip, LTO, size optimization flags, UPX). Acceptance: binary within 64K budget.
- [ ] **T-DEMO-5.7**: Implement 4K mode path (extreme minimization). Acceptance: binary within 4K budget (if achievable).
- [ ] **T-DEMO-5.8**: Verify no external dependencies at runtime (no asset files, no Python, no network). Acceptance: executable runs standalone on any compatible GPU.

**Dependencies**: Phase 3 (ray marching pipeline), Phase 2 (build-time compilation).
**Estimated Effort**: 2-3 weeks.

---

## Phase 6: Frame Graph Integration and Hybrid Rendering

- [ ] **T-DEMO-6.1**: Declare S13 ray march pass in S1 frame graph. Acceptance: S13 appears as a single compute pass in the frame graph.
- [ ] **T-DEMO-6.2**: Implement full-screen mode (S13 writes every pixel). Acceptance: pure demoscene scene renders without rasterization passes.
- [ ] **T-DEMO-6.3**: Implement hybrid mode depth buffer read. Acceptance: S13 reads rasterization depth buffer correctly.
- [ ] **T-DEMO-6.4**: Implement hybrid mode depth test (S13 writes only where closer). Acceptance: correct compositing of ray-marched and rasterized geometry.
- [ ] **T-DEMO-6.5**: Implement depth reconstruction from ray march hit distance. Acceptance: SDF hit distance converts to depth-buffer-compatible value.
- [ ] **T-DEMO-6.6**: Implement resource transitions between S13 and rasterization passes. Acceptance: 1-2 barriers per frame, no unnecessary transitions.
- [ ] **T-DEMO-6.7**: Implement multiple S13 passes (opaque + transparent). Acceptance: transparent SDF objects composite correctly over opaque SDF and raster geometry.
- [ ] **T-DEMO-6.8**: Verify S13 output feeds correctly into S8 post-processing. Acceptance: tone mapping, bloom, TAA work on S13 output.

**Dependencies**: Phase 3 (ray marching pipeline), S1 (frame graph), S8 (post-processing).
**Estimated Effort**: 2-3 weeks.

---

## Phase 7: Testing

### S13-A: SDF Primitive Correctness (~30 tests)

- [ ] **T-DEMO-7.1**: Implement and run 30 SDF primitive correctness tests (3-5 per primitive). Acceptance: each primitive returns correct signed distance at known sample points. Rotational invariance confirmed.

### S13-B: Combinator Correctness (~20 tests)

- [ ] **T-DEMO-7.2**: Implement and run 20 combinator correctness tests. Acceptance: union, intersection, subtraction, smooth variants, deep nesting, and material propagation all produce correct distance fields.

### S13-C: Ray Marching Pipeline (~15 tests)

- [ ] **T-DEMO-7.3**: Implement and run 15 ray marching pipeline tests. Acceptance: ray-sphere intersection, shadow correctness, AO, normal estimation, performance budget, WGSL vs Python reference match.

### S13-D: DSL Compiler (~25 tests)

- [ ] **T-DEMO-7.4**: Implement and run 25 DSL compiler tests. Acceptance: all primitives and combinators compile to valid WGSL, caching works, dirty invalidation works, optimizer produces correct output.

### S13-E: Texture-Free Materials (~15 tests)

- [ ] **T-DEMO-7.5**: Implement and run 15 material tests. Acceptance: FBM noise distribution, bump normal unit-length and direction, palette continuity, curvature detection, no NaN/Inf.

### S13-F: Size-Constrained Mode (~8 tests)

- [ ] **T-DEMO-7.6**: Implement and run 8 constraint mode tests. Acceptance: 64K compiles, binary size within budget, output is valid image, no external assets.

### Integration Tests (~10 tests)

- [ ] **T-DEMO-7.7**: Implement and run 10 integration tests. Acceptance: S13 writes to main framebuffer, hybrid compositing preserves both renderers, depth values consistent, post-processing works on S13-only framebuffer.

**Dependencies**: Phases 1-6 (test targets must exist).
**Estimated Effort**: 2-3 weeks.

---

## Phase 8: Algorithmic Research and Optimization

- [ ] **T-DEMO-8.1**: Research and implement analytic gradient propagation through combinators (approach: per-primitive gradient functions with winner-ID tracking). Acceptance: analytic normals match central differences within epsilon, with lower SDF evaluation cost.
- [ ] **T-DEMO-8.2**: Research and implement DSL optimization passes (pattern matching, CSE, automatic LOD for distant rays). Acceptance: compiled WGSL is smaller and/or faster than naive compilation for the same scene.
- [ ] **T-DEMO-8.3**: Research fractal SDF bounding (distance estimation ratio bailout, step count limits). Acceptance: fractal ray marching terminates reliably without getting trapped.
- [ ] **T-DEMO-8.4**: Research and implement importance-driven SDF evaluation (adaptive step count based on gradient magnitude). Acceptance: pixels with simple geometry receive fewer evaluations.
- [ ] **T-DEMO-8.5**: Research and implement temporal anti-aliasing for ray marching (reprojection via world-space hit position). Acceptance: stable image without motion vectors.
- [ ] **T-DEMO-8.6**: Research and implement automatic LOD for SDF scenes (reduced iteration depth with distance, simplified approximations for distant primitives). Acceptance: distant objects require fewer evaluations with acceptable visual quality.
- [ ] **T-DEMO-8.7**: Research bidirectional ray marching for SSS/translucency. Acceptance: light penetrates thin geometry with correct translucency.
- [ ] **T-DEMO-8.8**: Research DSL recompilation efficiency (incremental compilation strategies for WGSL). Acceptance: scene changes recompile in minimum time.

**Dependencies**: Phases 1-6 (research targets must exist).
**Estimated Effort**: Ongoing.

---

## Effort Summary

| Phase | Tasks | Est. Effort | Dependencies |
|-------|-------|-------------|-------------|
| 1: SDF Primitives & Combinators | 33 | 9-15 days | None |
| 2: Python DSL Compiler | 14 | 3-4 weeks | Phase 1 |
| 3: Ray March Pipeline | 13 | 3-4 weeks | Phase 1, S14, S3 |
| 4: Procedural Worlds & Materials | 16 | 3-4 weeks | Phase 3 |
| 5: 4K/64K Mode | 8 | 2-3 weeks | Phase 3, Phase 2 |
| 6: Frame Graph Integration | 8 | 2-3 weeks | Phase 3, S1, S8 |
| 7: Testing | 7 (test areas) | 2-3 weeks | Phases 1-6 |
| 8: Research & Optimization | 8 | Ongoing | Phases 1-6 |
| **Total** | **107** | **6-9 months** | |
