# TRINITY Archaeological Analysis

**Investigation Date:** 2026-05-22  
**Investigator:** Claude Code session  
**Branch:** task/T-BRG-0.1  
**Purpose:** Distinguish GRANDPHASE1 (original codebase) from GRANDPHASE2 (gap_sets-driven work)

---

## Executive Summary

TRINITY consists of two distinct development phases:

| Phase | Date | Description |
|-------|------|-------------|
| **GRANDPHASE1** | 2026-03-22 | One massive commit: 2,476 files, 1.1M lines — **entirely Python** |
| **GRANDPHASE2** | 2026-05-21+ | Gap_sets-driven work adding Rust backends (omega, renderer-backend) |

**Critical Finding:** GRANDPHASE1 was a complete Python game engine *framework* with null/stub backends. The actual GPU implementation (Vulkan, wgpu, etc.) was never built. GRANDPHASE2's gap_sets define the work to add real backends, but progress has been overstated due to Python fallbacks masking missing Rust code.

---

## Part 1: GRANDPHASE1 — "The Python Game Engine Framework"

### 1.1 Commit Analysis

```
Commit:   15ec548d260c234ed70544c68ce677c10708b82c
Author:   M <mstraughan86@hotmail.com>
Date:     Sun Mar 22 08:34:00 2026 -0400
Message:  first

Files:    2,476
Lines:    1,100,229 insertions
Languages: Python only (no Rust)
```

### 1.2 Directory Structure (GRANDPHASE1)

```
TRINITY/                           # 2,476 files, 1.1M lines
├── engine/                        # 987 files, 508K lines — 17 subsystems
│   ├── animation/                 # Skeletal, IK, motion matching, crowds
│   ├── audio/                     # Spatial, DSP, mixing, dialogue
│   ├── core/                      # ECS, math, memory, tasks, session
│   ├── debug/                     # Profiling, logging, replay, visual
│   ├── gameplay/                  # AI, abilities, camera, combat, quests
│   ├── integration/               # FlowForge, ShellLang, mods
│   ├── networking/                # Replication, RPC, lag compensation
│   ├── platform/                  # GPU/RHI, audio, input, window
│   ├── rendering/                 # Frame graph, GPU-driven, materials, lighting
│   ├── resource/                  # Streaming, virtualization, asset build
│   ├── simulation/                # Physics, cloth, fluid, destruction
│   ├── tooling/                   # Editor, hot-reload, level editor
│   ├── ui/                        # Widgets, layout, binding, accessibility
│   ├── world/                     # Terrain, foliage, HLOD, PCG
│   └── xr/                        # VR/AR runtime, avatars, spatial
│
├── trinity/                       # 121 files, 29K lines — Metaprogramming framework
│   ├── decorators/                # ~275 decorators (@component, @gpu, etc.)
│   ├── descriptors/               # Attribute access (RustStorage, Tracked)
│   ├── metaclasses/               # ComponentMeta, EngineMeta (~8 total)
│   └── tools/                     # Introspection, decompose()
│
├── foundation/                    # 25 files, 8K lines — Runtime infrastructure
│   ├── registry.py                # Type registry
│   ├── tracker.py                 # Dirty flag tracking
│   ├── eventlog.py                # Causal event chains
│   ├── mirror.py                  # Deep introspection
│   ├── bridge.py                  # Trinity ↔ Foundation connector
│   └── shelllang/                 # Interactive shell (5 primitives)
│
├── flowforge/                     # 585 files, 106K lines — Visual IDE
│   ├── apps/desktop/              # Tauri native shell
│   ├── flowforge_backend/         # Python sidecar (AST parser)
│   └── packages/                  # Shared TypeScript packages
│
├── tests/                         # 814 files, 425K lines — Test suite
│
└── docs/                          # 15 files — Original documentation
    ├── TOC.md                     # 48K — "Minimum Viable State-of-the-Art Game Engine"
    ├── TRINITY_LATEST.md          # 99K — Trinity pattern specification
    ├── GAME_ENGINE_INTEGRATION.md # Integration guide
    ├── GAME_ENGINE_INTEGRATION_TODO.md  # Original TODO (all [ ] not started)
    ├── PLATFORM_RHI_IMPLEMENTATION.md   # RHI with null backend design
    ├── FORMULAS.md
    ├── INTERCONNECTIVITY.md
    ├── QOL.md
    ├── TOC_FAST.md
    ├── VIPERIDE_v2.md
    ├── WINDOW_SUBSYSTEM.md
    ├── code_review_component_meta.md
    ├── code_review_metaclasses_report.md
    └── pcg_code_review.md
```

### 1.3 What Was Complete in GRANDPHASE1

| Component | Status | Evidence |
|-----------|--------|----------|
| Trinity Pattern | ✅ Complete | 989 tests passing, all 10 phases implemented |
| Foundation Runtime | ✅ Complete | Registry, Tracker, EventLog, Mirror, Bridge, ShellLang |
| FlowForge IDE | ✅ Complete | 585 files, Tauri + ComfyUI canvas |
| Engine Subsystems | ⚠️ Framework Only | Python data structures, no actual runtime |
| RHI Abstraction | ⚠️ Stubs Only | NullDevice, NullAdapter, NullBuffer, etc. |
| GPU Backends | ❌ Not Implemented | No Vulkan, no wgpu, no D3D12, no Metal |

### 1.4 The Original Architecture Philosophy

From `docs/PLATFORM_RHI_IMPLEMENTATION.md`:

> "All RHI components have fully functional null/stub implementations:
> - **Thread-safe**: All operations are thread-safe using locks
> - **Handle generation**: Unique handles for all resources
> - **Command recording**: Commands are recorded for validation/testing
> - **Fence simulation**: Proper wait/signal semantics with timeout support"

**Design Pattern:** Python abstractions with `try: import _omega except ImportError: pass` everywhere. Everything works without native code — native backends are purely optional optimization.

### 1.5 What Did NOT Exist in GRANDPHASE1

```
❌ crates/                    # No Rust code
❌ omega/                     # No deterministic math library
❌ docs/gap_sets/             # No roadmap phases
❌ Cargo.toml                 # No Rust workspace
❌ Any actual GPU execution   # Just null backends
```

---

## Part 2: GRANDPHASE2 — "The Rust Acceleration Layer"

### 2.1 Timeline

| Date | Event |
|------|-------|
| 2026-03-22 | GRANDPHASE1: "first" commit (1.1M lines) |
| 2026-03-22 to 2026-05-20 | **2-month gap** (0 commits) |
| 2026-05-21 | GRANDPHASE2 begins: T-CORE-*, T-DEMO-*, etc. |
| 2026-05-21 | Gap_sets recovered from session transcripts |
| 2026-05-22 | T-BRG-0.1 GREEN_LIGHT, CONCERNING_PROGRESS.md written |

### 2.2 New Directories (GRANDPHASE2)

```
crates/
├── renderer-backend/              # Frame graph IR, GPU buffers, WGSL
│   ├── Cargo.toml                 # wgpu, bytemuck, crossbeam, parking_lot, slotmap
│   ├── src/
│   │   ├── lib.rs                 # Exports frame_graph, gpu_driven, type_registry, bridge
│   │   ├── bridge.rs              # PyO3 placeholder (TODO stubs)
│   │   ├── type_registry.rs       # TypeRegistry, ComponentTypeInfo, FieldLayout
│   │   ├── component_store.rs     # Archetype SoA storage
│   │   └── frame_graph/
│   │       ├── mod.rs
│   │       └── python.rs          # PyPassNode → IrPass conversion
│   └── tests/
│       ├── blackbox_*.rs
│       └── whitebox_*.rs

omega/                             # Deterministic math library (Rust)
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── fixed.rs                   # Fixed16, Fixed32 (Q8.8, Q16.16)
│   ├── vec.rs                     # Vec2, Vec3, Vec4, FVec2, FVec3, FVec4
│   ├── mat.rs                     # Mat3, Mat4, M64 (column-major)
│   ├── quat.rs                    # Quat, FQuat
│   ├── rng.rs                     # SimRng (splitmix64 deterministic PRNG)
│   └── bridge.rs                  # PyO3 bindings (type_register, type_list, etc.)
└── tests/
    └── math_tests.rs

docs/gap_sets/                     # 20 roadmap phases (NEW)
├── GAPSET_1_CORE/
├── GAPSET_2_FRAME_GRAPH/
├── GAPSET_3_BRIDGE/               # ← ACTIVE
├── GAPSET_4_MATERIALS/
├── ... through ...
└── GAPSET_20_CROSS_CUTTING/
```

### 2.3 The 20 Gap Sets

| # | Gap Set | Domain | Status |
|---|---------|--------|--------|
| 1 | CORE | Deterministic math, fixed-point, Rust omega | Partial |
| 2 | FRAME_GRAPH | Render pass DAG, resource aliasing | Partial |
| **3** | **BRIDGE** | **Python↔Rust FFI via PyO3** | **ACTIVE** |
| 4 | MATERIALS | PBR pipeline, material DSL | Not started |
| 5 | LIGHTING | Direct/area lights, shadows | Not started |
| 6 | GI_REFLECTIONS | Global illumination, SSR | Not started |
| 7 | POST_PROCESS | Bloom, DOF, TAA | Partial |
| 8 | GPU_COMPUTE | Compute shaders, culling | Not started |
| 9 | RAY_TRACING | RT shadows, reflections | Not started |
| 10 | ENVIRONMENT | Sky, atmosphere, weather | Not started |
| 11 | DEMOSCENE | SDF, raymarching, procedural | Partial |
| 12 | ASSETS | glTF, streaming, virtualization | Not started |
| 13 | TOOLING | Editor, hot-reload | Partial |
| 14 | ANIMATION | Skeletal, IK, motion matching | Not started |
| 15 | AUDIO | Spatial, DSP, adaptive music | Partial |
| 16 | NETWORKING | Replication, rollback, RPC | Partial |
| 17 | GAMEPLAY | AI, abilities, quests | Not started |
| 18 | UI_XR | Widgets, VR/AR runtime | Not started |
| 19 | PHYSICS | Rigid body, cloth, destruction | Not started |
| 20 | CROSS_CUTTING | Determinism, profiling, security | Not started |

### 2.4 Task Naming Convention

```
T-{GAPSET}-{PHASE}.{NUMBER}

Examples:
  T-CORE-0.6    → GAPSET_1_CORE, Phase 0, Task 6
  T-BRG-1.4     → GAPSET_3_BRIDGE, Phase 1, Task 4
  T-DEMO-1.22   → GAPSET_11_DEMOSCENE, Phase 1, Task 22
  T-AU-2.14     → GAPSET_15_AUDIO, Phase 2, Task 14
  T-FG-1.1      → GAPSET_2_FRAME_GRAPH, Phase 1, Task 1
  T-PP-1.2      → GAPSET_7_POST_PROCESS, Phase 1, Task 2
  T-NET-1.7     → GAPSET_16_NETWORKING, Phase 1, Task 7
  T-GPU-1.6     → GAPSET_8_GPU_COMPUTE, Phase 1, Task 6
```

---

## Part 3: The Python Rendering Investigation

### 3.1 File Statistics

```
engine/rendering/
├── 55 Python files
├── 35,479 lines total
├── Largest files:
│   ├── postprocess_stack.py     1,776 lines
│   ├── material_graph.py        1,280 lines
│   ├── culling.py               1,109 lines
│   ├── material_functions.py    1,090 lines
│   └── particle_modules.py      1,016 lines
```

### 3.2 Subdirectories

| Directory | Files | Purpose |
|-----------|-------|---------|
| `framegraph/` | 10 | Pass declaration, DAG, barriers, async scheduling |
| `gpu_driven/` | 5 | Culling, bindless, visibility buffer |
| `materials/` | 6 | Material graph, shader compiler, PBR |
| `lighting/` | 5 | Shadows, GI (DDGI), shadow filtering |
| `postprocess/` | 8 | Bloom, DOF, TAA, upscaling, motion blur |
| `particles/` | 7 | GPU particles, trails, decals, VFX graph |
| `demoscene/` | 14 | SDF, WGSL codegen, raymarching |

### 3.3 What This Code Actually Is

**It is NOT working GPU rendering.**

It is:
1. **Data Structures** — PassNode, ResourceHandle, Barrier, MaterialNode
2. **Algorithms** — Frustum culling (pure Python math), DAG scheduling, resource aliasing
3. **Configuration Objects** — Quality presets, material parameters, light settings
4. **WGSL Code Generation** — Emits shader source strings (but doesn't compile/run them)

### 3.4 Critical Evidence: The Execute Method

From `engine/rendering/framegraph/frame_graph.py`:

```python
def _execute_barriers(
    self,
    batch: BarrierBatch,
    context: Any,
) -> None:
    """Execute a batch of barriers."""
    if batch.is_empty():
        return

    # In a real implementation, this would call into the RHI
    # to execute the actual GPU barriers.
    # The context object should provide a method like:
    #   context.execute_barriers(batch.barriers)
    # For now, we log/track the barriers for debugging purposes.
    if hasattr(context, 'execute_barriers'):
        context.execute_barriers(batch.barriers)
```

**The comment says it all:** "In a real implementation, this would call into the RHI"

### 3.5 The RHI: Null Backends Only

```
engine/platform/rhi/
├── device.py      → NullDevice, NullAdapter (stubs)
├── resources.py   → NullBuffer, NullTexture (stubs)
├── commands.py    → NullQueue, NullCommandList (stubs)
├── pipeline.py    → NullPipelineState (stub)
├── swapchain.py   → NullSwapchain (stub)
├── sync.py        → NullFence (stub)
├── binding.py     → NullDescriptorSet (stub)
├── raytracing.py  → Stub
└── mesh_shaders.py → Stub
```

### 3.6 Missing GPU Backends

The documentation (`PLATFORM_CONTEXT.md`) mentions:
```python
class VulkanDevice(GPUDevice): ...
class D3D12Device(GPUDevice): ...
```

**These classes do not exist.** Grep finds them only in documentation, not code.

### 3.7 Summary Table

| Layer | Exists? | Status |
|-------|---------|--------|
| Python Frame Graph | Yes | Data structures + algorithms (no GPU) |
| Python RHI Abstractions | Yes | Abstract base classes (ABC) |
| Null Backend | Yes | Stub implementations for testing |
| Vulkan Backend | **NO** | Never implemented |
| wgpu Backend | **NO** | Never implemented |
| D3D12 Backend | **NO** | Never implemented |
| Metal Backend | **NO** | Never implemented |
| Rust renderer-backend | Partial | Added in GRANDPHASE2, incomplete |

---

## Part 4: The Progress Reporting Problem

### 4.1 Why Progress Was Overstated

From `docs/CONCERNING_PROGRESS.md`:

1. **Python Fallbacks Mask Absence**: `RustStorageDescriptor` gracefully falls back to `StorageDescriptor`, so everything "works" even when Rust is absent
2. **Fabricated Checkmarks**: Many gap_set tasks were marked `[x]` complete when they weren't
3. **Independent Verification**: 77/204 BRIDGE items are REAL, 92 are ABSENT

### 4.2 The Fallback Pattern

```python
# From trinity/descriptors/rust_storage.py
def _make_storage_descriptor(name, default):
    try:
        from _omega import component_read, component_write
        return RustStorageDescriptor(name, default)
    except ImportError:
        # Fallback to pure Python - THIS ALWAYS SUCCEEDS
        return StorageDescriptor(name, default)
```

This means:
- Tests pass (Python fallback works)
- GREEN_LIGHT commits succeed
- But Rust code was never actually built/used

### 4.3 Detection Method

To detect if Rust is actually being used:
```python
import sys
if '_omega' in sys.modules:
    print("Rust bridge is ACTIVE")
else:
    print("Using Python fallback")
```

---

## Part 5: Current State Summary

### 5.1 What Is Complete

| Component | Status | Notes |
|-----------|--------|-------|
| Trinity Pattern | ✅ 100% | 989 tests, 10 phases |
| Foundation Runtime | ✅ 100% | All 6 systems |
| FlowForge IDE | ✅ 100% | Visual programming ready |
| Python ECS | ✅ Working | `engine/core/ecs/` |
| Python Math | ✅ Working | `engine/core/math/` |
| Test Infrastructure | ✅ Working | 814 test files |

### 5.2 What Is Partial

| Component | Status | Notes |
|-----------|--------|-------|
| omega crate | ~60% | Fixed-point, vectors, matrices exist; some operations missing |
| renderer-backend | ~30% | TypeRegistry, ComponentStore exist; frame graph partial |
| PyO3 Bridge | ~40% | type_register, component_read/write work; much TODO |

### 5.3 What Is Missing

| Component | Status | Needed For |
|-----------|--------|------------|
| GPU Backends | 0% | Actual rendering |
| Vulkan/wgpu | 0% | Graphics output |
| Physics Engine | 0% | Rigid body, cloth |
| Real Audio DSP | 0% | Spatial audio |
| Networking Runtime | 0% | Multiplayer |

---

## Part 6: Recommendations

### 6.1 Immediate Actions

1. **Audit All GREEN_LIGHT Commits**: Verify Rust code actually exists and is used
2. **Add Runtime Detection**: Log whether `_omega` is loaded or fallback is used
3. **Separate Tests**: Have tests that REQUIRE Rust (skip if not available)

### 6.2 Architectural Decisions Needed

1. **GPU Backend Choice**: Commit to wgpu (cross-platform) or Vulkan (performance)?
2. **Python/Rust Split**: What MUST be Rust vs what can stay Python?
3. **Bridge Granularity**: Component-level? Field-level? Batch operations?

### 6.3 Gap Set Priority

Based on dependencies:
1. GAPSET_3_BRIDGE (current) — Required for all Rust work
2. GAPSET_1_CORE — Math library completion
3. GAPSET_2_FRAME_GRAPH — Render orchestration
4. GAPSET_4_MATERIALS — First visual output
5. GAPSET_5_LIGHTING — Basic scene lighting

---

## Appendix A: Git History Analysis

### First 20 Commits (Chronological)

```
15ec548d 2026-03-22 first                                           # GRANDPHASE1
84726f22 2026-05-21 T-CORE-5.3a: RustStorageDescriptor              # GRANDPHASE2 begins
514da029 2026-05-21 T-CORE-5.3a: Wire into ComponentMeta
d7efcdd3 2026-05-21 T-CORE-2.6: Blackbox ECS tests
60cf8571 2026-05-21 T-CORE-5.3a: Blackbox RustStorageDescriptor
8d62692f 2026-05-21 T-CORE-5.3a: Whitebox RustStorageDescriptor
fbd75e01 2026-05-21 T-CORE-1.5 FIX: Slab allocator pool exhaustion
80146fd2 2026-05-21 T-CORE-1.5 FIX: Entity generation wrap
d5a68729 2026-05-21 T-CORE-3.4 FIX: TaskGraph.has_failures()
31f1de88 2026-05-21 T-CORE-5.3a FIX: _delete_stored
c16aec36 2026-05-21 T-CORE-5.3a FIX: byte offsets
932dc7e0 2026-05-21 T-CORE-2.6 FIX: DespawnCommand guard
...
7ff20ea9 2026-05-21 Recover ALL 20 gap_set PHASE_N_TODO.md files
...
c490827e 2026-05-22 docs: CONCERNING_PROGRESS.md
```

### Commit Statistics

| Metric | Value |
|--------|-------|
| Total commits | ~75 |
| GRANDPHASE1 commits | 1 |
| GRANDPHASE2 commits | ~74 |
| Days active | 2 (2026-05-21, 2026-05-22) |
| Gap between phases | 60 days |

---

## Appendix B: File Counts by Directory

```
Directory           Files    Lines    Language
-------------------------------------------------
engine/             987      508K     Python
tests/              814      425K     Python
flowforge/          307      106K     TS/Rust/Python
trinity/            121       29K     Python
crates/              30       19K     Rust
foundation/          25        8K     Python
omega/               10        5K     Rust
docs/                36       ~50K    Markdown
-------------------------------------------------
Total             ~2,330    ~1.15M
```

---

## Appendix C: Key Files Reference

### GRANDPHASE1 Original Docs
- `docs/TOC.md` — Engine reference (48K lines)
- `docs/TRINITY_LATEST.md` — Trinity spec (99K lines)
- `docs/GAME_ENGINE_INTEGRATION.md` — Integration guide
- `docs/GAME_ENGINE_INTEGRATION_TODO.md` — Original TODO list
- `docs/PLATFORM_RHI_IMPLEMENTATION.md` — RHI null backend design

### GRANDPHASE2 Gap Sets
- `docs/gap_sets/GAPSET_3_BRIDGE/PHASE_N_TODO.md` — Active work
- `docs/gap_sets/GAPSET_3_BRIDGE/CLARIFICATION.md` — Architectural decisions
- `docs/gap_sets/GAPSET_3_BRIDGE/GAP_3_SUMMARY.md` — Independent verification

### Key Source Files
- `trinity/metaclasses/component_meta.py` — Central metaprogramming
- `trinity/descriptors/rust_storage.py` — Python↔Rust descriptor
- `omega/src/bridge.rs` — PyO3 bindings
- `crates/renderer-backend/src/type_registry.rs` — Rust type system
- `engine/rendering/framegraph/frame_graph.py` — Python frame graph

---

## Appendix D: Complete Directory Structure

### D.1 engine/ — 18 Subsystems (508K lines)

| Directory | Files | Lines | Subdirectories |
|-----------|------:|------:|----------------|
| `engine/animation` | 68 | 40,045 | crowds, facial, graph, ik, motionmatching, procedural, skeletal, systems |
| `engine/audio` | 66 | 37,809 | adaptive, core, dialogue, dsp, mixing, spatial |
| `engine/common` | 4 | 0 | constants, types, utils |
| `engine/core` | 45 | 4,912 | ecs, math, memory, scheduler, session, tasks |
| `engine/debug` | 52 | 26,303 | console, crash, logging, profiling, replay, testing, tools, visual |
| `engine/determinism` | 5 | 0 | core, network, replay, snapshot |
| `engine/engine` | 5 | 0 | bootstrap, scheduler, session, world |
| `engine/gameplay` | 78 | 56,615 | abilities, ai, camera, combat, components, economy, entity, input, nav, quest |
| `engine/integration` | 7 | 0 | decorator_binding, descriptor_chain, flowforge, foundation_sync, mods, shelllang |
| `engine/networking` | 51 | 23,209 | lag_compensation, prediction, replication, rpc, security, serialization, social, tests, transport |
| `engine/platform` | 47 | 8,306 | audio, gpu, input, os, rhi, services, window |
| `engine/rendering` | 55 | 35,479 | demoscene, framegraph, gpu_driven, lighting, materials, particles, postprocess |
| `engine/resource` | 43 | 3,857 | asset, build, memory, streaming, types, virtualization |
| `engine/simulation` | 106 | 60,747 | character, cloth, collision, components, constraints, destruction, fluid, hair, physics, softbody, solver, vehicles |
| `engine/tooling` | 166 | 101,583 | animation_tools, assettools, automation, build, console, crash, debug, editor, hotreload, leveleditor, localization, logging, material_editor, profiling, replay, terrain, testing, undo, vcs, visual_scripting |
| `engine/ui` | 71 | 46,034 | accessibility, animation, binding, framework, layout, screens, styling, text, widgets |
| `engine/world` | 47 | 29,894 | environment, foliage, hlod, partition, pcg, queries, terrain |
| `engine/xr` | 60 | 33,129 | avatars, input, interaction, locomotion, platform, rendering, runtime, spatial, ui, utils |

### D.2 trinity/ — 4 Modules (28K lines)

| Directory | Files | Lines | Subdirectories |
|-----------|------:|------:|----------------|
| `trinity/decorators` | 72 | 19,757 | builtin_stacks |
| `trinity/descriptors` | 31 | 4,226 | — |
| `trinity/metaclasses` | 9 | 3,541 | — |
| `trinity/tools` | 5 | 256 | — |

### D.3 foundation/ — 2 Modules (8K lines)

| Directory | Files | Lines | Subdirectories |
|-----------|------:|------:|----------------|
| `foundation/` (root) | 20 | 6,587 | — |
| `foundation/shelllang` | 5 | 1,787 | — |

### D.4 Size Ranking (Top 15 by Lines)

| Rank | Directory | Lines | Notes |
|-----:|-----------|------:|-------|
| 1 | `engine/tooling` | 101,583 | Editor, hot-reload, VCS, 20 subdirs |
| 2 | `engine/simulation` | 60,747 | Physics, cloth, fluid, 12 subdirs |
| 3 | `engine/gameplay` | 56,615 | AI, abilities, combat, 10 subdirs |
| 4 | `engine/ui` | 46,034 | Widgets, layout, binding, 9 subdirs |
| 5 | `engine/animation` | 40,045 | Skeletal, IK, motion matching, 8 subdirs |
| 6 | `engine/audio` | 37,809 | Spatial, DSP, dialogue, 6 subdirs |
| 7 | `engine/rendering` | 35,479 | Frame graph, materials, 7 subdirs |
| 8 | `engine/xr` | 33,129 | VR/AR runtime, 10 subdirs |
| 9 | `engine/world` | 29,894 | Terrain, foliage, PCG, 7 subdirs |
| 10 | `engine/debug` | 26,303 | Profiling, replay, 8 subdirs |
| 11 | `engine/networking` | 23,209 | Replication, RPC, 9 subdirs |
| 12 | `trinity/decorators` | 19,757 | ~275 decorators |
| 13 | `engine/platform` | 8,306 | RHI, input, window, 7 subdirs |
| 14 | `foundation/` | 6,587 | Registry, Tracker, EventLog, etc. |
| 15 | `engine/core` | 4,912 | ECS, math, memory, 6 subdirs |

### D.5 Suspicious: Zero-Line Directories

| Directory | Files | Status | Investigation Needed |
|-----------|------:|--------|----------------------|
| `engine/common` | 4 | 0 lines | Empty stubs or __init__.py only? |
| `engine/determinism` | 5 | 0 lines | Planned but not implemented? |
| `engine/engine` | 5 | 0 lines | Bootstrap stubs? |
| `engine/integration` | 7 | 0 lines | FlowForge/ShellLang bridges? |

These directories have Python files but zero content lines — they may be:
- Empty `__init__.py` files only
- Stub files with only imports/docstrings
- Planned modules never implemented

---

## Appendix E: Deep Dive Targets

Based on the directory analysis, the following warrant detailed investigation:

### E.1 Priority 1: Core Infrastructure (Must Understand)

| Directory | Lines | Why |
|-----------|------:|-----|
| `engine/core/ecs` | ~1K | Entity-Component-System — heart of the engine |
| `engine/core/math` | ~1K | Math types — do they connect to omega? |
| `engine/platform/rhi` | ~2K | RHI abstractions — confirmed null backends |
| `trinity/metaclasses` | 3.5K | Class creation — ComponentMeta is central |
| `foundation/` | 6.6K | Runtime infrastructure — Registry, Tracker, etc. |

### E.2 Priority 2: Rendering Pipeline (GPU Path)

| Directory | Lines | Why |
|-----------|------:|-----|
| `engine/rendering/framegraph` | ~3K | Frame graph — confirmed "not real" |
| `engine/rendering/gpu_driven` | ~3K | GPU culling — pure Python math? |
| `engine/rendering/materials` | ~4K | Material system — shader generation? |
| `engine/rendering/demoscene` | ~5K | WGSL codegen — generates but doesn't run? |

### E.3 Priority 3: Large Subsystems (Unknown State)

| Directory | Lines | Why |
|-----------|------:|-----|
| `engine/tooling` | 101K | Largest subsystem — is editor functional? |
| `engine/simulation` | 61K | Physics — real implementation or stubs? |
| `engine/gameplay` | 57K | AI, abilities — connected to ECS? |

### E.4 Priority 4: Zero-Line Mysteries

| Directory | Files | Why |
|-----------|------:|-----|
| `engine/determinism` | 5 | Critical for netcode — why empty? |
| `engine/integration` | 7 | FlowForge bridge — why empty? |
| `engine/engine` | 5 | Bootstrap — why empty? |

---

*Document generated: 2026-05-22*
*Last updated: 2026-05-22 (added Appendix D, E)*
