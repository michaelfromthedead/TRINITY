# CLARIFICATION -- GAPSET_11_DEMOSCENE

## Mis-marked Tasks in PHASE_N_TODO.md

The following tasks are marked as completed [x] in PHASE_N_TODO.md but have NO corresponding implementation in the main source tree.

### Phase 1: Missing Implementation

| Task | Description | Marked | Reality |
|------|-------------|--------|---------|
| T-DEMO-1.11 | `sdf_octahedron(p, s)` | [x] | NO file found anywhere in repo. Not in engine/wgsl/ or crate/demoscene/. Not in ast_nodes.py SDF_PRIMITIVE_TYPE_MAP or wgsl_codegen.py _PRIMITIVE_TEMPLATES. |
| T-DEMO-1.12 | `sdf_pyramid(p, h)` | [x] | Same as above -- no file, no AST node, no codegen template. |
| T-DEMO-1.13 | `min2(a, b)` vec2 | [x] | No standalone WGSL combinator library. Codegen uses `select()` for pairwise material ID tracking but there is no `fn min2` anywhere. |
| T-DEMO-1.14 | `max2(a, b)` vec2 | [x] | Same -- no standalone implementation. |
| T-DEMO-1.15 | union combinator | [x] | No standalone implementation. Scene entry codegen combines primitives via select() but does NOT generate union/intersection/subtraction combinators. |
| T-DEMO-1.16 | intersection | [x] | Same -- no standalone implementation. |
| T-DEMO-1.17 | subtraction | [x] | Same -- no standalone implementation. |
| T-DEMO-1.18 | smooth union `smin` | [x] | No standalone implementation. |
| T-DEMO-1.19 | smooth intersection `smax` | [x] | Same. |
| T-DEMO-1.20 | smooth subtraction | [x] | Same. |
| T-DEMO-1.21 | displacement combinator | [x] | Same. |
| T-DEMO-1.30 | Perlin noise | [x] | noise_perlin.wgsl implemented at worktree `t-au-2.14-fix` but NOT in main source tree. |
| T-DEMO-1.31 | FBM noise | [x] | noise_fbm.wgsl implemented at worktree `t-au-2.14-fix` but NOT in main source tree. |

### Phase 1: Incorrectly Marked Incomplete

| Task | Description | Marked | Reality |
|------|-------------|--------|---------|
| T-DEMO-1.29 | Value noise | [ ] | noise_value.wgsl is FULLY IMPLEMENTED in main source with 1D, 2D, and 3D variants (125 lines). |

### Phase 2: Over-marked

| Task | Description | Marked | Reality |
|------|-------------|--------|---------|
| T-DEMO-2.1 | AST Builder | [x] | Implemented -- ast_nodes.py + ast_builder.py work correctly. |
| T-DEMO-2.2 | Trinity Pattern | [x] | __init__.py exports Mirror/Tracker patterns. Integration is shallow. |
| T-DEMO-2.3 | Primitive codegen | [x] | Generates 6 primitives (sphere, box, torus, cylinder, cone, plane, capsule). Need to add ellipsoid, box_frame, rounded_box. Missing octahedron, pyramid. |
| T-DEMO-2.4 | Combinator codegen | [ ] | NOT implemented. Codegen does not generate Union/Intersection/Subtraction/Smooth* calls. |
| T-DEMO-2.5 | Domain op codegen | [x] | Implemented and well-tested (all 6 domain ops with correct compensation). |
| T-DEMO-2.6 | Material codegen | [x] | Implemented with struct, switch/case, select() chain, tested thoroughly. |
| T-DEMO-2.7 | Scene codegen | [ ] | Scene entry point generated (sd_scene) but NO camera, lights, or render settings. |

## Architectural Notes

### Cylinder Convention Mismatch

The TODO spec says `sdf_cylinder(p, h)` where `h` is full height. The implementation in `sdf_cylinder.wgsl` and the codegen _SDF_CYLINDER_FN uses `h` as **half-height** (the Inigo Quilez convention). The generated codegen template `SDF_CYLINDER = "sdCylinder({position}, {height}, {radius})"` in the DSL passes h as-is. If the DSL user passes a full height, the cylinder will be twice as tall as expected. The WGSL shader and the codegen template are consistent with each other, but differ from the TODO spec's description.

### Cone Convention

The cone SDF uses 0-to-h along y (not centered), with r1 at bottom (y=0) and r2 at top (y=h). The codegen generates calls matching this convention. This is documented correctly in sdf_cone.wgsl.

### Plane SDF

The plane SDF takes parameters (p, normal, d) where d is the signed distance from origin to the plane. The implementation normalizes the normal internally. The plane in the TODO spec uses a 3-parameter form `sdf_plane(p, n)` with normal only, while the actual implementation uses 4 parameters `(p, n, d)`. The codegen matches the 4-param form.

### Scene Entry Point Name Convention

The codegen generates `sd_scene__{name}` for named scenes. This is not a standard convention; it uses double underscore to avoid collision with hand-written functions. The name derives from `SceneGraph.name` or the `name` parameter to `generate_wgsl()`.

## Test Coverage Notes

- 29 Python test files in `tests/rendering/demoscene/` covering SDF primitives (8 pairs), domain ops, noise (value/perlin/FBM), codegen (domain/material/ast_builder/scene)
- 2 Rust blackbox WGSL compilation tests in `.claude/worktrees/` (noise_perlin, noise_fbm)
- 1 Rust blackbox test in main source: `crates/renderer-backend/tests/blackbox_sdf_domain.rs`
- No integration tests, no real GPU execution tests, no 4K/64K tests
- Python tests all use reference implementations (no WGSL compilation in tests)
