# Phase 2: Shader Infrastructure — Architecture

## Status: PARTIAL

Shader infrastructure has the dependency graph (from GAP 3) and Python-side permutation infrastructure. Variant const system, include preprocessor, and Rust-side file watcher are absent.

## Current Architecture

### Rust: DepGraph (`material_dep_graph.rs`)

```
DepGraph
├── includes_to_materials: HashMap<String, Vec<u32>>
├── materials_to_includes: HashMap<u32, Vec<String>>
├── add_include(material_id, include)  → records bidirectional edge
└── invalidate(changed_include)        → BFS traversal, returns Vec<u32>
```

Key characteristics:
- Single-threaded (no RwLock)
- `invalidate()` is destructive — it removes edges during traversal
- Supports transitive invalidation through include chains
- 4 unit tests covering single, multi, unknown, and transitive cases

### Python: Permutation System (`shader_compiler.py`)

```
ShaderDefine { name, value? }         → Preprocessor define
PermutationKey { defines: FrozenSet } → Unique variant identifier
ShaderPermutation { defines, sources } → Compile variant set
  └── compile_variant(device)          → Returns CompiledShader
PSOCache { max_size, cache: OrderedDict } → PSO caching
```

### Python: Domain/Blend/Shading Enums (`material_system.py`)

```
MaterialDomain: SURFACE | DEFERRED_DECAL | VOLUME | POST_PROCESS | UI
BlendMode: OPAQUE | MASKED | TRANSLUCENT | ADDITIVE | MODULATE
ShadingModel: UNLIT | DEFAULT_LIT | SUBSURFACE | CLEAR_COAT | CLOTH | HAIR | EYE | FOLIAGE
```

### Python: Hot-Reload (`shader_compiler.py` + `engine/tooling/hotreload/`)

```
HotReloadWatcher
├── poll_interval: float
├── watched_files: Dict[Path, float]  → path → last_modified
└── check_for_changes()               → returns changed paths
```

## Missing for Functional Implementation

1. **Variant const system** — WGSL `const` bool declarations for domain/blend/quality gating
2. **Domain WGSL variants** — Gated code for SURFACE, DECAL, VOLUME, POST_PROCESS, UI
3. **Blend mode WGSL variants** — `discard` for MASKED, depth write skip for TRANSLUCENT
4. **Quality tier WGSL variants** — Light count limits, shadow quality, advanced shading gating
5. **Include preprocessor** — `#include` directive resolution with search paths and cycle detection
6. **Rust file watcher** — `notify` crate integration with debounce, DepGraph query, atomic PipelineTable swap

## Cross-References

- `crates/renderer-backend/src/material_dep_graph.rs` — Existing DepGraph (116 lines)
- `engine/rendering/materials/shader_compiler.py` — Python permutation system
- `engine/rendering/materials/material_system.py` — Domain/Blend/Shading enums
- `GAPSET_3_BRIDGE/GAP_3_SUMMARY.md` — Phase 8 DepGraph verification
