# TRINITY Design Documents Index

**Total Design Docs:** 464 architecture/design files  
**Total All Docs:** 886+ markdown files  
**Updated:** 2026-06-05

---

## Quick Status (READ FIRST)

| Document | Purpose |
|----------|---------|
| **[STATUS.md](STATUS.md)** | Current implementation state — single source of truth |
| **[PROGRESS_REPORT_2026_06_05.md](PROGRESS_REPORT_2026_06_05.md)** | Latest progress report with test metrics |
| **[SDLC_METHODOLOGY.md](SDLC_METHODOLOGY.md)** | Development methodology and pipeline |
| **[ARCHITECTURE_INVESTIGATION_REPORT.md](ARCHITECTURE_INVESTIGATION_REPORT.md)** | Complete architectural analysis (2026-05-24) |

**Summary:** Python 100% complete (600K lines). Rust 18% functional (140K lines exist). Bridge 100% complete. **Infrastructure complete (1370 tests): trinity-harness + trinity-contracts.** Rendering subsystems blocked on GAPSET_1_CORE completion.

---

## Infrastructure Crates (NEW)

| Crate | Tests | Purpose |
|-------|-------|---------|
| **trinity-harness** | 1147 | Multi-language test orchestration |
| **trinity-contracts** | 223 | Design-by-contract framework |

See `crates/trinity-harness/` and `crates/trinity-contracts/` for implementation.

---

## Authoritative Context Documents (SPECIFICATIONS)

These are the primary architecture references with Unreal-style design patterns:

| Document | Location | Lines | Key Content |
|----------|----------|-------|-------------|
| **RENDERING_CONTEXT.md** | `engine/rendering/` | 1,361 | Nanite visibility buffer, Lumen GI, 6 subsystems, all decorators |
| **TOOLING_CONTEXT.md** | `engine/tooling/` | 1,958 | FlowForge (Blueprint), material editor, profiler |
| **GAMEPLAY_CONTEXT.md** | `engine/gameplay/` | 1,192 | GAS abilities, behavior trees, utility AI |
| **RESOURCE_CONTEXT.md** | `engine/resource/` | --- | Asset pipeline, streaming |

**Unreal Engine 5 Parallels:**
- Nanite → `visibility_buffer.py`, `meshlet.py`
- Lumen → `gi_ddgi.py`, `gi_lumen.py` (planned)
- Niagara → `particles/vfx_graph.py`
- Blueprint → FlowForge
- GAS → `gameplay/abilities/`

---

## Document Categories

| Category | Count | Description |
|----------|-------|-------------|
| `RUST_DOCS/` | 215 | 20 gapsets with PHASE_N_ARCH, TODO, SUMMARY, PROJECT () |
| `PYTHON_DOCS/` | 608 | Detailed subsystem investigations () |
| `evaluations/` | 32 | Code quality and reality assessments |
| `reviews/` | 3 | Code review reports |
| `archive/historical/` | 7 | Historical docs (completed work) |
| Root docs | 22 | Top-level architecture and planning |

---

## Gap Sets (20 Gapsets)

### GAPSET_1_CORE (10 docs)
- `GAP_1_SUMMARY.md`
- `PHASE_1_ARCH.md` - Core ECS architecture
- `PHASE_2_ARCH.md` - Memory subsystem
- `PHASE_3_ARCH.md` - Threading/jobs
- `PHASE_4_RHI_ARCH.md` - RHI abstraction
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_2_FRAME_GRAPH (11 docs)
- `GAP_2_SUMMARY.md`
- `PHASE_1_ARCH.md` - Frame graph IR
- `PHASE_5_ARCH.md` - Async scheduling
- `PHASE_7_ARCH.md` - Resource aliasing
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_3_BRIDGE (18 docs)
- `GAP_3_SUMMARY.md`
- `PHASE_7_COMMAND_CHANNEL_FRAME_GRAPH_ARCH.md`
- `PHASE_9_FULL_FEATURES_ARCH.md`
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_4_MATERIALS (9 docs)
- `GAP_4_SUMMARY.md`
- `PHASE_1_ARCH.md` - Material graph
- `PHASE_2_ARCH.md` - Shader compilation
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_5_LIGHTING (10 docs)
- `GAP_5_SUMMARY.md`
- `PHASE_2_FROXEL_ARCH.md` - Clustered froxel lighting
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_6_GI_REFLECTIONS (15 docs)
- `GAP_6_SUMMARY.md`
- `PHASE_1_ARCH.md` - SH/probe foundation
- `PHASE_2_ARCH.md` - DDGI core
- `PHASE_3_ARCH.md` - SSGI
- `PHASE_7_ARCH.md` - Voxel GI
- `PHASE_8_ARCH.md` - RT reflections
- `PHASE_9_ARCH.md` - Denoising
- `PHASE_10_ARCH.md` - Visualization
- `PHASE_11_ARCH.md` - Research
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_7_POST_PROCESS (10 docs)
- `GAP_7_SUMMARY.md`
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_8_GPU_COMPUTE (11 docs)
- `GAP_8_SUMMARY.md`
- `PHASE_3_ARCH.md` - Compute pipelines
- `PHASE_7_ARCH.md` - Advanced compute
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_9_RAY_TRACING (7 docs)
- `GAP_9_SUMMARY.md`
- `PHASE_1_ARCH.md` - BLAS/TLAS
- `PHASE_2_ARCH.md` - RT pipelines
- `PHASE_3_ARCH.md` - Path tracing
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_10_ENVIRONMENT (7 docs)
- `GAP_10_SUMMARY.md`
- `PHASE_1_ARCH.md` - Sky/atmosphere
- `PHASE_2_ARCH.md` - Weather
- `PHASE_3_ARCH.md` - Time of day
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_11_DEMOSCENE (12 docs)
- `GAP_11_SUMMARY.md`
- `PHASE_1_ARCH.md` through `PHASE_8_ARCH.md`
- `PHASE_N_TODO.md`
- `PROJECT.md`
- `CLARIFICATION.md`

### GAPSET_12_ASSETS (6 docs)
- `PHASE_1_ARCH.md` - Asset pipeline
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_13_TOOLING (13 docs)
- Multiple PHASE_ARCH docs
- `PHASE_N_TODO.md`
- `CLARIFICATION.md`

### GAPSET_14_ANIMATION (13 docs)
- Multiple PHASE_ARCH docs
- `PHASE_N_TODO.md`

### GAPSET_15_AUDIO (13 docs)
- Multiple PHASE_ARCH docs
- `PHASE_N_TODO.md`

### GAPSET_16_NETWORKING (14 docs)
- `PHASE_1_ARCH.md` through `PHASE_10_ARCH.md`
- `PHASE_N_TODO.md`
- `CLARIFICATION.md`

### GAPSET_17_GAMEPLAY (14 docs)
- `PHASE_5_ARCH.md`
- `PHASE_N_TODO.md`

### GAPSET_18_UI_XR (4 docs)
- `PHASE_N_TODO.md`

### GAPSET_19_PHYSICS (10 docs)
- `PHASE_3_ARCH.md`
- `PHASE_N_TODO.md`
- `PROJECT.md`

### GAPSET_20_CROSS_CUTTING (5 docs)
- `PHASE_N_TODO.md`

---

## Phase Output (608 docs in 50+ subdirectories)

### Rendering Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_rendering_gpu_driven/` | 13 | INVESTIGATION, PHASE_2_ARCH, PHASE_3_ARCH, PHASE_4_ARCH, SUMMARY |
| `engine_rendering_lighting/` | 13 | INVESTIGATION, PHASE_3_ARCH, PHASE_4_ARCH, SUMMARY |
| `engine_rendering_materials/` | 17 | INVESTIGATION, PHASE_6_ARCH |
| `engine_rendering_framegraph/` | 15 | INVESTIGATION, SUMMARY |
| `engine_rendering_particles/` | 13 | INVESTIGATION, CLARIFICATION |
| `engine_rendering_postprocess/` | 11 | PHASE_1_TODO |
| `engine_rendering_demoscene/` | 10 | Multiple PHASE docs |

### Simulation Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_simulation_physics_solver/` | 14 | PHASE_2_ARCH |
| `engine_simulation_character_cloth_collision/` | 15 | PHASE_3_ARCH |
| `engine_simulation_components_constraints_softbody_vehicles/` | 15 | PHASE_3_ARCH |
| `engine_simulation_destruction_fluid_hair/` | 15 | PHASE_2_ARCH, PHASE_3_ARCH |

### Animation Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_animation_skeletal_systems/` | 17 | Multiple PHASE docs |
| `engine_animation_graph_ik/` | 16 | Multiple PHASE docs |
| `engine_animation_motionmatching_procedural/` | 15 | Multiple PHASE docs |
| `engine_animation_crowds_facial/` | 15 | Multiple PHASE docs |

### Audio Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_audio_adaptive_core/` | 17 | PHASE_2_ARCH |
| `engine_audio_mixing_spatial/` | 18 | Multiple PHASE docs |
| `engine_audio_dialogue_dsp/` | 12 | Multiple PHASE docs |

### Gameplay Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_gameplay_abilities_ai_camera/` | 21 | CLARIFICATION |
| `engine_gameplay_economy_entity_input/` | 19 | Multiple PHASE docs |
| `engine_gameplay_nav_quest/` | 16 | INVESTIGATION, PHASE_1_ARCH, PHASE_3_ARCH |
| `engine_gameplay_combat_components/` | 11 | Multiple PHASE docs |

### UI Subsystems
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_ui_widgets/` | 13 | SUMMARY |
| `engine_ui_accessibility_animation_binding_framework/` | 16 | PHASE_3_ARCH, PHASE_4_ARCH |
| `engine_ui_layout_screens_styling_text/` | 15 | Multiple PHASE docs |

### Platform/Infrastructure
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `engine_platform/` | 24 | engine_platform_rhi.md |
| `engine_networking/` | 21 | Multiple PHASE docs |
| `engine_tooling/` | 35 | PHASE_3_ARCH, leveleditor, build |
| `engine_debug_resource/` | 19 | INVENTORY, EVALUATIONS, PHASE_1_ARCH, PHASE_2_ARCH |
| `engine_world/` | 18 | PHASE_2_ARCH, PHASE_3_ARCH, hlod, foliage, pcg |
| `engine_xr/` | 27 | Multiple PHASE docs |

### Foundation/Core
| Directory | Docs | Key Files |
|-----------|------|-----------|
| `foundation/` | 14 | Multiple docs |
| `trinity_decorators_part1/` | 17 | Multiple PHASE docs |
| `trinity_decorators_part2/` | 16 | Multiple PHASE docs |
| `trinity_descriptors_metaclasses/` | 13 | Multiple PHASE docs |

### Standalone Files
- `GRAND_SYNTHESIS.md` - Complete 600K line verification
- `RUST_BACKLOG.md` - Rust implementation backlog
- `engine_core_*.md` - Core subsystem docs (6 files)
- `engine_common_*.md` - Common types docs (3 files)
- `engine_determinism_*.md` - Determinism docs (4 files)
- `engine_engine_*.md` - Engine bootstrap docs (4 files)
- `engine_integration_*.md` - Integration docs (6 files)
- `engine_resource_*.md` - Resource system docs (6 files)

---

## Evaluations (32 docs)

| File | Description |
|------|-------------|
| `SUMMARY.md` | Overall evaluation summary |
| `rust/SUMMARY.md` | Rust crate evaluation |
| `rust/frame_graph.md` | Frame graph evaluation |
| `rendering_misc.md` | Rendering evaluation |
| + 28 more subsystem evaluations |

---

## Root-Level Architecture Docs

| File | Description |
|------|-------------|
| **`STATUS.md`** | **Current implementation status — single source of truth** |
| **`architecture/ARCHITECTURE_INVESTIGATION_REPORT.md`** | **Complete architectural analysis (2026-05-24)** |
| `architecture/RENDERER_ARCHITECTURE.md` | Renderer guide (references CONTEXT.md) |
| `investigations/WGPU_INVESTIGATION.md` | wgpu codebase investigation |
| `investigations/WGPU_EXECUTION_PLAN.md` | 5-phase wgpu wiring plan |
| `investigations/RUST_INVESTIGATIONS.md` | Rust backend investigation |
| `archive/historical/ARCHAEOLOGICAL_ANALYSIS.md` | Deep codebase archaeology |
| `GRAND_SYNTHESIS.md` | 600K line verification |
| `TOC.md` | Table of contents |
| `TOC_FAST.md` | Quick reference TOC |
| `PLATFORM_RHI_IMPLEMENTATION.md` | RHI implementation details |
| `GAME_ENGINE_INTEGRATION.md` | Engine integration guide |
| `specs/FORMULAS.md` | Mathematical formulas |
| `specs/INTERCONNECTIVITY.md` | Module interconnections |

---

## Key Design Documents to Read

For understanding the FULL renderer architecture:

### Must Read
1. `PYTHON_DOCS/GRAND_SYNTHESIS.md` - The 600K line reality check
2. `RUST_DOCS/GAPSET_6_GI_REFLECTIONS/PROJECT.md` - GI/reflection design
3. `RUST_DOCS/GAPSET_9_RAY_TRACING/PROJECT.md` - Ray tracing design
4. `PYTHON_DOCS/engine_rendering_gpu_driven/INVESTIGATION.md` - Visibility buffer
5. `PYTHON_DOCS/engine_rendering_lighting/INVESTIGATION.md` - Lighting pipeline

### Per Subsystem (in RUST_DOCS/)
- Frame Graph: `GAPSET_2_FRAME_GRAPH/`
- Materials: `GAPSET_4_MATERIALS/`
- Lighting: `GAPSET_5_LIGHTING/`
- GI: `GAPSET_6_GI_REFLECTIONS/`
- Post-Process: `GAPSET_7_POST_PROCESS/`
- GPU Compute: `GAPSET_8_GPU_COMPUTE/`
- Ray Tracing: `GAPSET_9_RAY_TRACING/`
