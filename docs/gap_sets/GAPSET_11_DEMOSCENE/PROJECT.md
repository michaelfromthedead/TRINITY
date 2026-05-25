# GAPSET_11_DEMOSCENE -- Project Overview

## Goal

Build a complete demoscene rendering subsystem within the Trinity engine, supporting signed-distance-field (SDF) ray marching via WGSL compute shaders, a Python domain-specific language (DSL) for authoring scenes, and size-constrained (4K/64K) standalone executables for demoscene competitions.

## Architecture (8 Phases)

```
Phase 1: SDF Primitives & Combinators  (WGSL Library)
    --> phase_1_arch.md

Phase 2: Python SDF DSL Compiler        (Python -> WGSL)
    --> phase_2_arch.md

Phase 3: Ray Marching Compute Pipeline  (WGSL Compute Shader)
    --> phase_3_arch.md

Phase 4: Procedural Worlds & Materials  (WGSL Noise + SDF)
    (no arch doc -- no implementation)

Phase 5: 4K/64K Size-Constrained Mode   (Rust + WGSL)
    (no arch doc -- no implementation)

Phase 6: Frame Graph Integration        (Rust + WGSL)
    (no arch doc -- no implementation)

Phase 7: Testing                        (Python + Rust)
    (no arch doc -- no implementation)

Phase 8: Research & Optimization        (Ongoing)
    (no arch doc -- no implementation)
```

## Dependency Chain

```
Phase 1 (SDF primitives/noise/domain ops)
    |
    v
Phase 2 (DSL compiler targeting Phase 1 WGSL)
    |
    v
Phase 3 (Ray marching consuming Phase 1 SDF)
    |
    +---+---+
    v       v
Phase 4   Phase 5   Phase 6
    |       |
    +---+---+
        v
    Phase 7 (Testing all phases)
    
Phase 8 (Ongoing research, depends on 1-6)
```

## Key File Locations

| Area | Path |
|------|------|
| Crate WGSL (domain, noise hash, noise value) | `crates/renderer-backend/src/demoscene/` |
| SDF primitive WGSL shaders (10 files) | `engine/rendering/demoscene/wgsl/` |
| Python DSL compiler (4 modules) | `engine/rendering/demoscene/` |
| Python tests (29 files) | `tests/rendering/demoscene/` |
| Rust domain blackbox tests | `crates/renderer-backend/tests/` |

## Current State

- **Phase 1**: 10/12 SDF primitives implemented; 6/6 domain ops implemented; hash/value noise implemented; Perlin/FBM in worktrees only. Combinators, octahedron, pyramid NOT implemented in main source.
- **Phase 2**: AST builder + WGSL codegen for primitives/domain/materials implemented. Combinator codegen, scene codegen, optimization passes, cache, error reporting NOT implemented.
- **Phases 3-8**: Zero implementation in main source.
