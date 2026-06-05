# RUST vs PYTHON Documentation Map

**Created:** 2026-05-25  
**Purpose:** Comprehensive breakdown of which documentation corresponds to Rust backend vs Python frontend  
**Mission:** Enable newcomers to instantly understand the documentation structure and navigate to the right files

---

## The Rule

```
docs/RUST_DOCS/*     = RUST backend implementation tasks (ACTIVE)
docs/PYTHON_DOCS/* = PYTHON frontend investigation/analysis (ACTIVE)
docs/evaluations/*  = PYTHON codebase health (+ rust/ subfolder for Rust)
engine/*/CONTEXT.md = SPECIFICATION (language-agnostic, defines both)
```

---

## Documentation Categories

### 1. RUST Documentation

#### `docs/RUST_DOCS/` — Rust Implementation Roadmap (ACTIVE)
**215 files, 34,868 lines**

| Directory | Focus | Status |
|-----------|-------|--------|
| GAPSET_1_CORE | Math, memory, ECS, task system | 49% |
| GAPSET_2_FRAME_GRAPH | Render pass DAG, barriers, wgpu commands | 28% |
| GAPSET_3_BRIDGE | PyO3 Python↔Rust FFI | **100%** ✅ |
| GAPSET_4_MATERIALS | PBR pipeline, material DSL | 6% |
| GAPSET_5_LIGHTING | Direct/area lights, shadows | 3% |
| GAPSET_6_GI_REFLECTIONS | Global illumination, SSR | 0% |
| GAPSET_7_POST_PROCESS | Bloom, DOF, TAA | 29% |
| GAPSET_8_GPU_COMPUTE | Compute shaders, culling | 34% |
| GAPSET_9_RAY_TRACING | RT shadows, reflections | 9% |
| GAPSET_10_ENVIRONMENT | Sky, atmosphere, weather | 0% |
| GAPSET_11_DEMOSCENE | SDF, raymarching, procedural | 43% |
| GAPSET_12_ASSETS | glTF, streaming | 0% |
| GAPSET_13_TOOLING | Editor hot-reload | 39% |
| GAPSET_14_ANIMATION | Rust animation backend | 65% |
| GAPSET_15_AUDIO | Rust audio backend | 71% |
| GAPSET_16_NETWORKING | Rust netcode | 69% |
| GAPSET_17_GAMEPLAY | Rust gameplay | 88% |
| GAPSET_18_UI_XR | Rust UI/XR | 80% |
| GAPSET_19_PHYSICS | Rust physics | 65% |
| GAPSET_20_CROSS_CUTTING | Determinism, profiling | 50% |

**Each gapset contains:**
- `PROJECT.md` — Overview and goals
- `PHASE_N_TODO.md` — Task checklist with [x]/[~]/[-] markers
- `PHASE_*_ARCH.md` — Architecture documents
- `CLARIFICATION.md` — Design decisions
- `GAP_*_SUMMARY.md` — Status summary

#### `docs/evaluations/rust/` — Rust Codebase Health
**6 files**

| File | Evaluates |
|------|-----------|
| SUMMARY.md | Overview of Rust backend state |
| omega.md | omega crate (math library) |
| frame_graph.md | Frame graph IR and compiler |
| gpu_driven.md | GPU tables (material, mesh, texture) |
| memory_ecs.md | Memory management and ECS |
| rhi_wgpu.md | RHI and wgpu integration |

#### Rust-Focused Root Docs

| File | Lines | Purpose |
|------|-------|---------|
| RUST_INVESTIGATIONS.md | 28,918 | Deep Rust codebase analysis |
| WGPU_INVESTIGATION.md | 13,960 | wgpu integration research |
| WGPU_EXECUTION_PLAN.md | 16,103 | 5-phase wgpu wiring plan |
| RENDERER_BACKEND_CLEANUP.md | 12,267 | Rust renderer cleanup tasks |
| RENDERER_ARCHITECTURE.md | 14,010 | Rendering pipeline design |

---

### 2. PYTHON Documentation

#### `docs/PYTHON_DOCS/` — Python Subsystem Investigation (ACTIVE)
**608 files, 128,692 lines**

Organized by engine module. Each directory contains deep analysis of Python implementation.

| Directory Pattern | Covers |
|-------------------|--------|
| `engine_core_*` | ECS, math, memory, scheduler, session, tasks |
| `engine_rendering_*` | Frame graph, GPU-driven, lighting, materials, particles, postprocess, demoscene |
| `engine_simulation_*` | Physics, solver, cloth, fluid, destruction, vehicles, character, hair |
| `engine_animation_*` | Skeletal, IK, motion matching, crowds, facial, procedural |
| `engine_audio_*` | Core, spatial, DSP, mixing, dialogue, adaptive |
| `engine_gameplay_*` | AI, abilities, combat, camera, economy, navigation, quest |
| `engine_networking` | Replication, prediction, RPC, lag compensation |
| `engine_platform` | RHI ABCs, windowing, input |
| `engine_ui_*` | Widgets, layout, accessibility, binding |
| `engine_world` | Terrain, foliage, PCG, HLOD |
| `engine_xr` | VR/AR runtime, avatars |
| `engine_tooling` | Editor, hot-reload, visual scripting |
| `engine_resource_*` | Asset pipeline, streaming, virtualization |
| `engine_debug_*` | Profiling, logging, replay |
| `foundation` | Mirror, Serializer, Registry, Tracker |
| `trinity_*` | Metaclasses, decorators, descriptors |

**Key Files:**
- `GRAND_SYNTHESIS.md` — High-level Python findings
- `INVESTIGATION_TODO.md` — Investigation task list
- `RUST_BACKLOG.md` — Python→Rust migration backlog

#### `docs/evaluations/` (root) — Python Module Health
**24 reports + SUMMARY.md**

| Report | Evaluates |
|--------|-----------|
| trinity.md | Metaclasses, decorators, descriptors |
| foundation.md | Runtime infrastructure |
| engine_core.md | Engine loop, ECS, math |
| simulation_physics.md | Physics + solver |
| simulation_cloth.md | Cloth simulation |
| simulation_character.md | Character + hair |
| simulation_misc.md | Collision, destruction, vehicles |
| rendering_framegraph.md | Frame graph |
| rendering_postprocess.md | Post-processing |
| rendering_misc.md | Lighting, materials, particles |
| animation.md | Animation pipeline |
| audio.md | Audio system |
| gameplay_ai.md | AI, navigation |
| gameplay_misc.md | Combat, quest, economy |
| networking.md | Multiplayer stack |
| platform.md | Platform abstractions |
| resource.md | Asset pipeline |
| ui.md | UI framework |
| world.md | World systems |
| xr.md | VR/AR |
| debug.md | Debug tools |
| tooling.md | Editor suite |
| empty_scaffolding.md | Directories to delete |
| test_suite.md | Test coverage analysis |

#### Python-Focused Root Docs

| File | Lines | Purpose |
|------|-------|---------|
| TRINITY_LATEST.md | 99,439 | Trinity Pattern full spec |
| PYTHON_EVALUATION_TODO.md | 12,238 | 24-unit evaluation plan |
| PYTHON_VERSION_PLAN.md | 4,917 | Python 3.13 requirement |
| code_review_metaclasses_report.md | 12,273 | SystemMeta/ResourceMeta review |
| code_review_component_meta.md | 10,868 | ComponentMeta review |
| pcg_code_review.md | 6,071 | PCG code review |
| SKIPPED_TESTS_DEBT.md | 10,411 | Test debt tracking |

---

### 3. SPECIFICATION (Both Languages)

#### `engine/*/CONTEXT.md` — Authoritative Specs
**14 files, 17,054 lines**

These define WHAT to build (language-agnostic). Python implements the algorithms, Rust implements the GPU dispatch.

| File | Lines | Defines |
|------|-------|---------|
| TOOLING_CONTEXT.md | 1,968 | FlowForge (Blueprint), Editor |
| SIMULATION_CONTEXT.md | 1,473 | Physics, collision, vehicles |
| RESOURCE_CONTEXT.md | 1,450 | Asset pipeline, streaming |
| XR_CONTEXT.md | 1,378 | VR/AR, hand tracking |
| RENDERING_CONTEXT.md | 1,360 | Nanite, Lumen, DDGI |
| CORE_CONTEXT.md | 1,310 | ECS, memory, math, scheduling |
| PLATFORM_CONTEXT.md | 1,256 | RHI/wgpu, windowing, input |
| DEBUG_CONTEXT.md | 1,201 | Profiling, hot-reload |
| GAMEPLAY_CONTEXT.md | 1,191 | GAS abilities, behavior trees |
| WORLD_CONTEXT.md | 1,111 | World partition, streaming |
| NETWORKING_CONTEXT.md | 973 | Netcode, replication |
| ANIMATION_CONTEXT.md | 860 | IK, motion matching |
| AUDIO_CONTEXT.md | 744 | Spatial audio, DSP |
| UI_CONTEXT.md | 647 | Widgets, accessibility |

---

### 4. CROSS-CUTTING Documentation

| File | Lines | Category | Purpose |
|------|-------|----------|---------|
| ARCHITECTURE_INVESTIGATION_REPORT.md | 22,314 | BOTH | Complete architectural analysis |
| ARCHAEOLOGICAL_ANALYSIS.md | 25,753 | BOTH | GRANDPHASE1 vs GRANDPHASE2 history |
| STATUS.md | 5,606 | BOTH | Current implementation state |
| DESIGN_DOCS_INDEX.md | 10,513 | BOTH | 462 document catalogue |
| GAME_ENGINE_INTEGRATION.md | 85,562 | BOTH | Integration guide |
| GAME_ENGINE_INTEGRATION_TODO.md | 36,473 | BOTH | Original TODO list |
| TOC.md | 48,394 | BOTH | Engine reference |
| TOC_FAST.md | 33,639 | BOTH | Quick reference |
| FORMULAS.md | 51,570 | BOTH | Mathematical foundations |
| INTERCONNECTIVITY.md | 7,697 | BOTH | System dependencies |
| REMAINING_WORK_ROADMAP.md | 11,105 | BOTH | Prioritized work |
| CONCERNING_PROGRESS.md | 4,595 | BOTH | Progress reporting issues |

---

## Quick Reference

### "I want to understand Python code"
```
docs/PYTHON_DOCS/  # Deep analysis per module (ACTIVE)
docs/evaluations/*.md                 # Health reports (not rust/)
docs/specs/TRINITY_LATEST.md                # Trinity Pattern spec
docs/reviews/                         # Code reviews
```

### "I want to understand Rust code"
```
docs/RUST_DOCS/      # Implementation tasks (ACTIVE)
docs/evaluations/rust/                # Rust health reports
docs/investigations/RUST_INVESTIGATIONS.md           # Deep Rust analysis
docs/investigations/WGPU_*.md                        # wgpu-specific
docs/architecture/RENDERER_*.md                    # Renderer-specific
```

### "I want to understand the spec"
```
engine/*/CONTEXT.md         # Authoritative specifications
```

### "I want to understand the project"
```
docs/STATUS.md              # Current state
docs/ARCHITECTURE_INVESTIGATION_REPORT.md  # Complete analysis
docs/ARCHAEOLOGICAL_ANALYSIS.md            # History
README.md                   # Quick overview
```

---

## Statistics

| Category | Files | Lines | Location |
|----------|-------|-------|----------|
| **Rust Docs** | 221+ | ~60,000 | RUST_DOCS/, evaluations/rust/, RUST_*.md, WGPU_*.md |
| **Python Docs** | 632+ | ~180,000 | PYTHON_DOCS/, evaluations/, reviews/ |
| **Specs** | 14 | 17,054 | engine/*/CONTEXT.md |
| **Cross-Cutting** | 15+ | ~300,000 | Root docs/ files |
| **TOTAL** | ~880 | ~550,000+ | docs/ |

---

## The Two Grand Phases

| Phase | Date | Focus | Documentation |
|-------|------|-------|---------------|
| **GRANDPHASE1** | 2026-03-22 | Python framework (1.1M lines) | phase_output/, TRINITY_LATEST.md |
| **GRANDPHASE2** | 2026-05-21+ | Rust backend wiring | gap_sets/, RUST_*.md, WGPU_*.md |

---

*This document is the map. Use it to navigate.*
