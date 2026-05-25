# GAPSET_4_MATERIALS — Architectural Clarifications

## Architectural Philosophy: DSL vs Node-Based Material Authoring

The GAPSET_4_MATERIALS todo spec assumes a **Python-DSL-driven** material authoring model: users write Python classes with a `surface()` method, and an AST->WGSL compiler translates these to WGSL shader code.

The actual codebase has **two competing material authoring systems**:

### 1. DSL System (`trinity/materials/`) — Planned but Incomplete

- `dsl.py` provides `Material`, `SurfaceContext`, `SurfaceOutput` base classes
- `compiler.py` has a `MaterialCompiler` class, but `_walk()` returns a static placeholder
- No MaterialMeta metaclass, no AST->WGSL translation
- **Status:** Scaffold only. 3 files, ~50 lines total non-stub code

### 2. Node-Based Graph System (`engine/rendering/materials/` + `engine/tooling/material_editor/`) — Functional

- `material_graph.py` provides 25+ node types (Add, Multiply, Lerp, TextureSample, etc.)
- `GraphCompiler` converts the node graph to shader source code
- `material_editor/material_compiler.py` compiles to HLSL/GLSL/Metal (NOT WGSL)
- `material_editor/material_nodes.py` has 54KB of node definitions
- Material editor has a full UI with preview, library, instances, parameters
- **Status:** Rich and functional, but targets HLSL/GLSL/Metal, not WGSL

### Key Divergence

The spec assumes a single DSL->WGSL pipeline. The codebase has a Python-native material system that uses a node graph approach and targets traditional shading languages. This means:

1. **No WGSL output from the material editor** — it generates HLSL, GLSL, or Metal
2. **The existing PBR WGSL shaders** (`pbr.frag.wgsl`, `pbr.vert.wgsl`) are hand-written, not compiler-generated
3. **Two paths need reconciliation**: either (a) add WGSL output to the node graph compiler, (b) build the DSL->WGSL compiler as planned, or (c) use the hand-written WGSL shaders and drive them via the bindless material table

## Bindless Table Pattern vs Uniform-Based PBR

The spec describes `PBRInput`/`PBRParams`/`PBROutput` WGSL structs. The actual codebase uses a **bindless material table** pattern:

- `gpu_driven/material_table.rs` — `MaterialTableEntry` (80 bytes) in a GPU storage buffer
- `pbr.frag.wgsl` reads `material_table[input.material_index]` for all PBR parameters
- This is a **more flexible** approach than per-pipeline uniforms — material changes are just an index change, not a pipeline swap

**Impact on Phase 1 (PBR template):** The template should generate WGSL that reads from `material_table[mat_idx]` rather than defining standalone `PBRParams` structs. The bindless table already exists and is tested.

## Python-Native ECS vs Rust Bridge

The spec assumes a Rust+PyO3 bridge for material registration (e.g., `material_register()` in `bridge.rs`). The actual codebase has:

- `bridge.rs` — TODO comments only, no actual PyO3 functions
- `omega/Cargo.toml` — the PyO3 omega crate exists separately
- Python material system operates entirely in Python without Rust bridge calls

**This means material compilation, parameter validation, and editor operations all happen in pure Python.** The Rust side only gets involved for GPU operations (material table upload, pipeline compilation, rendering).

## WGSL Shader Maturity

The PBR WGSL shaders (`pbr.frag.wgsl`, `pbr.vert.wgsl`) are surprisingly complete for a project at this stage:

| Feature | Status |
|---------|--------|
| Cook-Torrance BRDF (GGX, Smith, Schlick) | Complete |
| Directional/Point/Spot lights | Complete |
| CSM shadows with PCF | Complete |
| Emissive + Ambient terms | Complete |
| Bindless material table | Complete — differs from spec structs |
| Normal mapping | Placeholder (commented "would go here") |
| Variant gating | Absent |
| Advanced shading | Absent |
| Alpha mask/discard | Absent |

## Gap Priority Assessment

### High Priority (Blocks everything else)
1. **AST->WGSL compiler** (T-MAT-1.2) — Without this, the DSL is decorative
2. **PBR pipeline integration** (T-MAT-3.4) — Without this, PBR shaders are untested
3. **Variant const system** (T-MAT-2.1) — Required for quality tiers, domain variants

### Medium Priority (Unlocks material workflow)
4. **Shader include system** (T-MAT-2.5) — Enables reusable WGSL modules
5. **File watcher + hot-reload** (T-MAT-2.7) — Development iteration speed
6. **Material animation** (T-MAT-5.5) — Time-driven parameters are standard
7. **Material inheritance** (T-MAT-5.2) — Material reuse

### Low Priority (Production hardening)
8. All Phase 6-10 (Content Store, Mesh, Texture, Asset pipelines)
9. Phase 11 hardening

## Actual vs Spec'd Architecture

```
SPEC'D ARCHITECTURE:                    ACTUAL ARCHITECTURE:

Python DSL class                        trinity/materials/dsl.py
  ↓ AST->WGSL compiler                    (stub — no compiler)
  ↓                                     trinity/materials/compiler.py
WGSL PBR template                         (stub — returns placeholder)
  ↓                                     ╔════════════════════════════╗
PBRInput/PBRParams/PBROutput            ║ pbr.frag.wgsl + pbr.vert  ║
  ↓                                     ║ (hand-written, complete)    ║
ShaderCache → PipelineTable             ║ ShaderCache → PipelineTable ║
  ↓                                     ║ gpu_driven/material_table  ║
Frame Graph Pass                        ║ renderer.rs (triangle only)║
  ↓                                     ╚════════════════════════════╝
wgpu Render
                                        engine/rendering/materials/
                                          (Python node-based system)
                                              → HLSL/GLSL/Metal output
                                              → NOT WGSL
```

## Cross-References to GAPSET_3_BRIDGE

| GAP 3 Item | GAP 4 Dependency | Status |
|---|---|---|
| T-BRG-6.2 (PBR WGSL) | Phase 3 (all) | REAL — shaders complete |
| T-BRG-4.1 (PipelineTable) | T-MAT-3.4, 6.5 | REAL — but no LRU, no sharding |
| T-BRG-8.3 (DepGraph) | T-MAT-2.6 | REAL — single-threaded, tested |
| T-BRG-5.1 (Mesh/Material tables) | T-MAT-5.7, 8.x, 9.x | REAL — bindless tables |
| T-BRG-7.2 (Frame Graph) | T-MAT-3.4 | REAL — 1681 lines IR |
| T-BRG-8.1 (DSL scaffold) | T-MAT-1.1, 1.2 | PARTIAL — scaffold only |
| T-BRG-8.2 (Shader compiler) | T-MAT-1.3, 1.7 | DIVERTED — Python materials |
| T-BRG-2.1 (ComponentStore) | T-MAT-10.7 (parameter channel) | ABSENT — Python ECS only |
| T-BRG-10.1 (Memory) | Phase 6-7 (content store) | ABSENT — Python memory |
