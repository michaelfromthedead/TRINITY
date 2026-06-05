# PHASE 8: Command Channel -- Material DSL + Hot Reload

**Scope:** Provide a Python-embedded material description language (DSL) that compiles to WGSL, with a dependency graph for hot-reload of shader sources.
**Depends on:** Phase 6 (PBR shaders as compilation target), Phase 7 (frame graph for pipeline recreation)
**Produces:** Material DSL Python module (AST-to-WGSL compiler), DependencyGraph for hot-reload, PyO3 material_register()
**Status:** DIVERTED -- The original TODO specified an AST-based DSL (Python AST walker -> WGSL output). The actual implementation uses a node-based graph material editor (`engine/tooling/material_editor/`) connected to a shader compiler (`shader_compiler.py`). No MaterialMeta metaclass or SurfaceContext/SurfaceOutput DSL exists. The hot-reload system is fully implemented in Python (`engine/tooling/hotreload/` with 6 modules).

## 1. Overview

Phase 8 covers two related but distinct concerns: how materials are authored (DSL or graph editor), and how they are compiled and hot-reloaded at runtime. The original plan called for a Python DSL where material functions are decorated with `@material` and compiled to WGSL via AST traversal. The actual implementation took a graph-based approach: nodes with input/output sockets connected in a visual editor, compiled to a shader by `shader_compiler.py`. Both approaches are valid; the divergence is one of UX philosophy, not engineering quality.

## 2. Architectural decisions

- **Graph-based material editing instead of DSL**: The material editor (`engine/tooling/material_editor/material_graph.py`, `material_nodes.py`, `node_factory.py`) implements a node graph with typed sockets, connections, and compilation. Users connect nodes visually instead of writing DSL code. The compiler produces WGSL from the graph topology.
- **ShaderCompiler as the compilation backend**: `engine/rendering/materials/shader_compiler.py` takes a material graph description and emits WGSL. It handles node ordering, variable naming, and shader stage assignment. This exists and works.
- **Hot-reload is Python-only**: The hot-reload system (`engine/tooling/hotreload/`) watches file system changes via `module_watcher.py`, tracks dependencies via `dependency_tracker.py`, and triggers callbacks via `reload_callbacks.py`. The Rust side has no PipelineTable yet, so no atomic pipeline swap mechanism exists.
- **No Rust-side material registration**: Unlike the Type Channel (Phase 1) which has a Rust TypeRegistry stub, Phase 8 has no Rust material registry. `material_register()` was never implemented.

## 3. Constraints specific to this phase

- The shader compiler must produce valid WGSL output that passes naga validation.
- Hot-reload must not cause visual artifacts: the old pipeline must continue rendering until the new one is fully compiled, then atomically swap.
- Material parameters must match the MaterialTable WGSL struct (defined in gpu_driven/material_table.wgsl). The compiler should reject materials whose parameter types don't match.
- Python file watchers are OS-dependent: `inotify` on Linux, `kqueue` on macOS, `ReadDirectoryChangesW` on Windows.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `trinity/materials/dsl.py` | Python-embedded material DSL | DOES NOT EXIST (original approach) |
| `trinity/materials/compiler.py` | AST-to-WGSL compiler | DOES NOT EXIST (original approach) |
| `engine/rendering/materials/material_graph.py` | Node-based material graph | EXISTS (actual approach) |
| `engine/rendering/materials/material_system.py` | Material management system | EXISTS |
| `engine/rendering/materials/pbr_model.py` | Python PBR reference material | EXISTS |
| `engine/rendering/materials/shader_compiler.py` | Graph-to-WGSL compiler | EXISTS |
| `engine/tooling/material_editor/material_compiler.py` | Editor-to-compiler bridge | EXISTS |
| `engine/tooling/material_editor/material_graph.py` | Editor graph data model | EXISTS |
| `engine/tooling/material_editor/material_nodes.py` | Node type library | EXISTS |
| `engine/tooling/material_editor/node_factory.py` | Node registry | EXISTS |
| `engine/tooling/hotreload/dependency_tracker.py` | File dependency graph | EXISTS |
| `engine/tooling/hotreload/hot_reload.py` | Hot reload orchestrator | EXISTS |
| `engine/tooling/hotreload/module_watcher.py` | File change watcher | EXISTS |
| `engine/tooling/hotreload/reload_callbacks.py` | Post-reload callbacks | EXISTS |
| `engine/tooling/hotreload/state_preservation.py` | Pipeline state preservation | EXISTS |
| `engine/tooling/hotreload/schema_hash.py` | Shader schema hashing | EXISTS |
| Rust PipelineTable (pipeline.rs) | Compiled pipeline cache | DOES NOT EXIST |
| Rust atomic pipeline swap | Tear-down/creation synchronization | DOES NOT EXIST |
| `bridge.rs` PyO3 material_register() | Register materials from Python | STUB |

## 5. Testing strategy

- Unit: Graph-to-WGSL compilation round-trip (shader_compiler.py).
- Unit: MaterialTable struct alignment between Rust/WGSL and Python output.
- Integration: Hot-reload workflow -- edit a material file, verify the dependency tracker detects the change, shader recompiles, pipeline swaps without visible artifacts.
- Integration: Material editor -> compiler -> wgpu pipeline end-to-end with a simple unlit material.

## 6. Open questions

- Should the DSL approach be abandoned entirely in favor of the graph editor? The graph editor is more work to build but is more accessible to non-programmer artists. The DSL is simpler to implement and test. A hybrid approach (graph editor compiles to a DSL representation) is possible.
- The Python hot-reload system is complete and tested. Is a Rust-side hot-reload (PipelineTable + atomic swap) needed, or can the Python system handle all hot-reload scenarios? Python can rebuild the frame graph, but Rust-side pipeline compilation needs a trigger.

## 7. References

- Phase 6 (PBR) defines the shader output that the material compiler targets.
- Phase 7 (Frame Graph) handles pipeline recreation after hot-reload.
- GAP_3_SUMMARY.md section "Phase 8: Material DSL" (corrected status: 6 real, 4 partial, 13 absent -- marked "DIVERTED").
