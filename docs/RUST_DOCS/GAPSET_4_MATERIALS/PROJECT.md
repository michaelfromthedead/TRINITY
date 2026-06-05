# GAPSET_4_MATERIALS — Project Overview

## Scope

The material system for TRINITY's rendering pipeline. Covers the full material authoring-to-rendering stack: DSL-based material definition, WGSL shader generation, PBR rendering, variant compilation, content addressed storage, mesh and texture pipelines, and asset streaming.

## Goals

1. **DSL-Driven Material Authoring** — Define materials in Python with an AST->WGSL compiler translating surface() functions to WGSL
2. **PBR Core Rendering** — Physically-based shading with Cook-Torrance BRDF, full light loop, CSM shadows
3. **Variant System** — Compile-time const bool gating for domain, blend mode, and quality tier variants
4. **Material System** — Templates, instances, inheritance, animation, LOD, bindless resources
5. **Content Store** — Content-addressed storage with structural sharing, streaming, delta sync, GC
6. **Mesh Pipeline** — glTF loading, meshlet generation, BLAS, LOD, budget-aware selection
7. **Texture Pipeline** — Importer plugins, format selection, cooking, virtual texturing
8. **Asset Pipeline** — Predictive pre-loading, priority queues, caching, hot-reload, streaming heuristics
9. **Hardening** — E2E tests, visual regression, benchmarks, memory audit, cross-platform validation

## Phase Overview

| Phase | Name | Status | Core Deliverables |
|-------|------|--------|-------------------|
| 1 | DSL Foundation | NOT STARTED | MaterialMeta metaclass, AST->WGSL compiler, PBR template, builtins, texture binding, SurfaceContext |
| 2 | Shader Infrastructure | NOT STARTED | Variant const system, domain/blend/quality variants, include preprocessor, DepGraph, file watcher |
| 3 | PBR Core | MOSTLY DONE | WGSL PBR structs, Cook-Torrance BRDF, light loop, shadows, pipeline integration |
| 4 | Advanced Shading | PYTHON ONLY | SSS, clear coat, anisotropy, sheen, transmission, iridescence (Python models only) |
| 5 | Material System | PARTIAL | Rich Python material system; missing WGSL domain impls, animation, LOD |
| 6 | Content Store Foundation | NOT STARTED | ContentHash, FileBackend, ContentTree, BLAKE3, pipeline sharding |
| 7 | Content Store Advanced | NOT STARTED | Streaming, diffing, sync, GC, provenance pruning |
| 8 | Mesh Pipeline | NOT STARTED | glTF loader, meshlets, BLAS, LOD gen, budget tracking |
| 9 | Texture Pipeline | NOT STARTED | Importers, cooking, virtual texturing, cubemaps, format importers |
| 10 | Asset Pipeline | NOT STARTED | Pre-loading, priority queues, caching, hot-reload, heuristics |
| 11 | Hardening | NOT STARTED | E2E tests, visual regression, benchmarks, audit, cross-platform |

## Architecture

### Rust Layer (`crates/renderer-backend/`)

```
src/
├── pipeline.rs              # ShaderCache + PipelineTable (SHA-256 dedup)
├── material_dep_graph.rs    # DepGraph (BFS invalidation, single-threaded)
├── renderer.rs              # wgpu Renderer (triangle only, no PBR mesh)
├── frame_graph/
│   ├── mod.rs               # Frame Graph IR (1681 lines)
│   └── python.rs            # PyO3 bridge for frame graph
├── gpu_driven/
│   ├── material_table.rs    # Bindless MaterialTable (80B entries)
│   ├── material_table.wgsl  # WGSL MaterialTableEntry + helpers
│   ├── mesh_table.rs        # Bindless MeshTable
│   ├── texture_table.rs     # TextureTable
│   └── buffers.rs           # Triple-buffered GPU staging
shaders/
├── pbr.frag.wgsl            # Cook-Torrance BRDF, light loop, CSM shadows
└── pbr.vert.wgsl            # PBR vertex transform
```

### Python Layer

```
trinity/materials/
├── __init__.py               # Re-exports
├── dsl.py                    # Material/SurfaceContext/SurfaceOutput stubs
└── compiler.py               # Stub returning placeholder WGSL

engine/rendering/materials/
├── __init__.py               # Full material system exports
├── material_system.py        # MaterialTemplate, MaterialInstance, domains, blend modes
├── pbr_model.py              # PBRParameters, PBRMaterial, PBRTextureSet
├── shader_compiler.py        # ShaderSource, Permutation, PSOCache, HotReloadWatcher
├── material_graph.py         # 25+ node types, GraphCompiler
├── material_functions.py     # 14 reusable shader functions
├── advanced_models.py        # All 6 advanced shading models
└── constants.py              # PBR ranges, shader constants, SSS params

engine/tooling/material_editor/
├── material_compiler.py      # Graph->HLSL/GLSL/Metal compilation
├── material_graph.py         # Editor graph types
├── material_nodes.py         # Node definitions (54KB)
├── node_factory.py           # Node factory
├── material_library.py       # Material library management
├── material_instances.py     # Instance overrides
├── material_preview.py       # Real-time preview
├── material_parameters.py    # Parameter types
└── connection_validator.py   # Type-safe connection validation
```

## Cross-References

| Reference | Description |
|-----------|-------------|
| GAPSET_3_BRIDGE | Built PBR WGSL shaders, PipelineTable, DepGraph, bindless material table |
| GAPSET_3_BRIDGE/PHASE_8_COMMAND_CHANNEL_MATERIAL_DSL_ARCH.md | Material DSL architecture |
| GAPSET_3_BRIDGE/GAP_3_SUMMARY.md | Full GAP 3 verification report |
