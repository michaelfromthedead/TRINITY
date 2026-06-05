# PHASE 7 ARCH: Testing

## Status: PARTIAL (tests exist for Phase 1-2, Phase 3-6 tests NOT implemented)

Phase 7 has 7 test areas defined with ~108 total tests. Only Phase 1 and Phase 2 code has associated tests.

## Existing Test Files

### Python tests in `tests/rendering/demoscene/` (29 files)

| Test | Tasks Covered | Type |
|------|---------------|------|
| test_sdf_sphere_whitebox/blackbox | T-DEMO-1.1 | Python ref model |
| test_sdf_box_whitebox/blackbox | T-DEMO-1.2 | Python ref model |
| test_sdf_torus_whitebox/blackbox | T-DEMO-1.3 | Python ref model |
| test_sdf_cylinder_whitebox | T-DEMO-1.4 | Python ref model |
| test_sdf_ellipsoid_whitebox/blackbox | T-DEMO-1.8 | Python ref model |
| test_sdf_plane_whitebox/blackbox | T-DEMO-1.6 | Python ref model |
| test_sdf_box_frame_whitebox/blackbox | T-DEMO-1.9 | Python ref model |
| test_sdf_capsule_fix_whitebox/blackbox | T-DEMO-1.7 | Python ref model |
| test_sdf_rounded_box_whitebox/blackbox | T-DEMO-1.10 | Python ref model |
| test_sdf_domain_whitebox | T-DEMO-1.22-1.27 | Python ref model |
| test_sdf_domain_twist_blackbox | T-DEMO-1.25 | Python ref model |
| test_noise_perlin_whitebox/blackbox | T-DEMO-1.30 | Python ref model |
| test_noise_value_whitebox | T-DEMO-1.29 | Python ref model |
| test_noise_fbm_whitebox | T-DEMO-1.31 | Python ref model |
| test_ast_builder/whitebox | T-DEMO-2.1 | Builder output |
| test_domain_codegen_whitebox/blackbox | T-DEMO-2.5 | WGSL template strings |
| test_material_codegen_whitebox/blackbox | T-DEMO-2.6 | WGSL output |
| test_scene_codegen_blackbox | T-DEMO-2.7 | WGSL output |

### Rust tests in `crates/renderer-backend/tests/`

| Test | Tasks Covered | Type |
|------|---------------|------|
| blackbox_sdf_domain.rs | T-DEMO-1.22-1.27 | naga WGSL compilation |
| blackbox_noise_fbm.rs (worktree only) | T-DEMO-1.31 | naga WGSL compilation |
| blackbox_noise_perlin.rs (worktree only) | T-DEMO-1.30 | naga WGSL compilation |

## Missing Tests (per PHASE_N_TODO.md)

| Test Area | Tasks | Required | Existing | Missing |
|-----------|-------|----------|----------|---------|
| S13-A: SDF Primitive Correctness | T-DEMO-7.1 | ~30 tests | ~18 (blackbox + whitebox) | ~12 (octahedron, pyramid, combinators, Ray Marching pipeline) |
| S13-B: Combinator Correctness | T-DEMO-7.2 | ~20 tests | 0 | 20 |
| S13-C: Ray Marching Pipeline | T-DEMO-7.3 | ~15 tests | 0 | 15 |
| S13-D: DSL Compiler | T-DEMO-7.4 | ~25 tests | ~5 (domain/mat/scene codegen) | ~20 (combinator codegen, optimizations, cache, error reporting) |
| S13-E: Texture-Free Materials | T-DEMO-7.5 | ~15 tests | 0 | 15 |
| S13-F: Size-Constrained Mode | T-DEMO-7.6 | ~8 tests | 0 | 8 |
| Integration | T-DEMO-7.7 | ~10 tests | 0 | 10 |
