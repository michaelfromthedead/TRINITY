# GAPSET_11_DEMOSCENE -- Summary

## Scope

Demoscene/SDF rendering subsystem: signed distance functions, ray marching, noise functions, Python DSL compiler for SDF scenes, and size-constrained (4K/64K) mode.

## TASK_IDs: 46 total (Phase 1-7), scope defined in PHASE_N_TODO.md

## Files On Disk

### Crate WGSL (renderer-backend/src/demoscene/)
| File | Task | Lines | Status |
|------|------|-------|--------|
| `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/demoscene/noise_hash.wgsl` | T-DEMO-1.28 | 110 | Fully implemented in main source |
| `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/demoscene/noise_value.wgsl` | T-DEMO-1.29 | 125 | Fully implemented in main source |
| `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/demoscene/sdf_domain.wgsl` | T-DEMO-1.22-1.27 | 217 | 6 domain operations fully implemented |

### Engine WGSL (engine/rendering/demoscene/wgsl/)
| File | Task | Lines | Status |
|------|------|-------|--------|
| `sdf_sphere.wgsl` | T-DEMO-1.1 | 39 | Implemented |
| `sdf_box.wgsl` | T-DEMO-1.2 | 52 | Implemented |
| `sdf_torus.wgsl` | T-DEMO-1.3 | 35 | Implemented |
| `sdf_cylinder.wgsl` | T-DEMO-1.4 | 43 | Implemented (half-height) |
| `sdf_cone.wgsl` | T-DEMO-1.5 | 72 | Implemented (truncated frustum) |
| `sdf_plane.wgsl` | T-DEMO-1.6 | 44 | Implemented |
| `sdf_capsule.wgsl` | T-DEMO-1.7 | 45 | Implemented |
| `sdf_ellipsoid.wgsl` | T-DEMO-1.8 | 56 | Implemented |
| `sdf_box_frame.wgsl` | T-DEMO-1.9 | 49 | Implemented |
| `sdf_rounded_box.wgsl` | T-DEMO-1.10 | 58 | Implemented |

### Missing WGSL (not in main source, only in worktrees)
| File | Task | Reality |
|------|------|---------|
| `noise_perlin.wgsl` | T-DEMO-1.30 | Exists ONLY in worktree `t-au-2.14-fix` and `t-demo-1.10-fix` |
| `noise_fbm.wgsl` | T-DEMO-1.31 | Exists ONLY in worktree `t-au-2.14-fix` |
| `sdf_octahedron.wgsl` | T-DEMO-1.11 | NOT FOUND anywhere |
| `sdf_pyramid.wgsl` | T-DEMO-1.12 | NOT FOUND anywhere |

### Python DSL (engine/rendering/demoscene/)
| File | Task | Lines | Status |
|------|------|-------|--------|
| `ast_nodes.py` | T-DEMO-2.1 | 245 | SceneGraph, 10+ node types |
| `ast_builder.py` | T-DEMO-2.1, 2.2 | 242 | Dict/lambda/object AST builder |
| `wgsl_codegen.py` | T-DEMO-2.3, 2.5, 2.6 | 646 | SDF+domain+material codegen |
| `__init__.py` | Re-export | 16 | Module exports |

### Test Files (main tests/rendering/demoscene/)
29 Python test files and 2 Rust test files (blackbox_sdf_domain.rs) across:
- 8 SDF primitive whitebox/blackbox test pairs (sphere, box, torus, cylinder, box_frame, ellipsoid, plane, capsule_fix, rounded_box)
- 1 SDF domain whitebox + 1 blackbox (twist)
- 1 noise_perlin whitebox + 1 blackbox
- 1 noise_value whitebox
- 1 noise_fbm whitebox
- 1 domain_codegen whitebox + 1 blackbox
- 1 material_codegen whitebox + 1 blackbox
- 1 ast_builder whitebox + 1 blackbox
- 2 Rust blackbox tests (blackbox_noise_fbm.rs, blackbox_noise_perlin.rs in worktree)

## Key Findings

### 1. Octahedron and Pyramid SDFs are missing
T-DEMO-1.11 and T-DEMO-1.12 are marked [x] but no implementation files exist anywhere in the repository. The `SDF_PRIMITIVE_TYPE_MAP` in ast_nodes.py (line 226-234) does NOT include entries for octahedron or pyramid. The WGSL codegen primitives dict in wgsl_codegen.py (line 156-164) does NOT include sdOctahedron or sdPyramid.

### 2. Combinators have NO standalone WGSL library
T-DEMO-1.13 through T-DEMO-1.21 are all marked [x], but no standalone `sdf_combinators.wgsl` file exists. The codegen generates `min2`-style logic inline via `select()`, but there is no separately compilable WGSL combinator library with `fn min2`, `fn max2`, `fn smin`, `fn smax`, `fn sdf_displaced`.

### 3. Perlin noise and FBM exist ONLY in worktrees
noise_perlin.wgsl and noise_fbm.wgsl are implemented (110 and 161 lines respectively) and have Rust blackbox tests, but they exist ONLY in the `.claude/worktrees/t-au-2.14-fix/` directory. They have NOT been promoted to the main source tree.

### 4. Value noise is mis-tagged in TODO
noise_value.wgsl is fully implemented (1D, 2D, 3D) but T-DEMO-1.29 is marked [ ] (incomplete).

### 5. Python DSL codegen is extensive but incomplete
- AST builder works with dict markers, lambda disassembly, and DSL objects
- WGSL codegen covers sphere/box/torus/cylinder/cone/plane/capsule (but not ellipsoid, box_frame, rounded_box octahedron, pyramid)
- Material codegen fully covered with struct + switch/case + select() chain
- Scene entry point generates sd_scene() combining pipeline + primitives + compensation
- Combinator codegen is NOT implemented (codegen generates no Union/Intersection/Subtraction/Smooth* calls)
- Camera/light/render settings codegen is NOT implemented
- No optimization passes (constant folding, DCE, CSE, domain repetition flattening, material merging)
- No cached compilation with dirty invalidation
- No error reporting

## Conclusions

The demoscene subsystem has substantial WGSL implementation for Phase 1 primitives and domain operations, and a working Python DSL toolkit. However, several Phase 1 tasks are mis-marked (octahedron/pyramid/combinators/Perlin/FBM are not actually in main source), and Phases 3-8 have zero implementation on disk.
