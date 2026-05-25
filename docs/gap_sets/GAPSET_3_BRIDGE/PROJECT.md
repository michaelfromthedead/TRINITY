# GAPSET_3_BRIDGE -- Python-Rust Bridge for TRINITY Engine

**Owner:** Michael
**Status:** COMPLETE (39/39 tasks GREEN_LIGHT)
**RDC run:** retroactive, 2026-05-22
**Completed:** 2026-05-22

## 1. Goal

Build a bidirectional bridge between Python (the TRINITY engine's rapid-iteration layer) and Rust (its GPU-accelerated, safety-critical core). The bridge operates over three logical channels: a **Type Channel** that registers Python component schemas with Rust, a **Data Channel** that routes ECS storage reads and writes through Rust's SoA (Struct of Arrays) component store, and a **Command Channel** that streams rendering commands from Python into the wgpu-based GPU renderer. A companion GPU math library (the `omega` crate) provides Fixed32/f32 types with bytemuck `Pod`/`Zeroable` guarantees for direct GPU upload.

## 2. Why

Python delivers ergonomics and rapid iteration for engine logic, editor tooling, and asset pipelines. Rust delivers deterministic numerics, memory safety, and GPU compute access through wgpu. The bridge allows gradual, feature-by-feature activation -- Python code continues working at every step while Rust components are wired in behind PyO3 gates. This is not a rewrite; it is a progressive acceleration layer.

## 3. Hardware / Environment constraints

- Target: GPU-accelerated desktop (wgpu abstracts Vulkan/Metal/DX12/WebGPU)
- Rust edition 2021
- Python 3.12+
- bytemuck Pod/Zeroable required for all GPU-uploadable types
- Cross-platform: Windows/Linux/macOS (wgpu handles backend abstraction)

## 4. Non-goals

- Not replacing the Python ECS -- augmenting it with Rust storage
- Not a standalone engine -- always part of TRINITY
- Not a Vulkan/DirectX/Metal-specific backend -- wgpu abstracts all three
- Not a full Rust rewrite of the engine -- only the GPU-critical path

## 5. Phase overview

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| 0 | Crate Scaffolding | ✅ COMPLETE | Workspace with omega + renderer-backend |
| 1 | Type Channel Protocol | ✅ COMPLETE | TypeRegistry, PyO3 type_register/type_list, ComponentMeta wiring |
| 2 | Data Channel — Component Store | ✅ COMPLETE | SoA ComponentStore, RustStorageDescriptor, World→Rust routing |
| 3 | Data Channel — GPU Math Library | ✅ COMPLETE | omega crate: Vec/Mat/Quat/Fixed/AABB/Frustum/Ray, 316 tests |
| 4 | Command Channel — Triangle in wgpu | ✅ COMPLETE | Renderer skeleton, triangle pipeline, 5 PyO3 command functions |
| 5 | Data Channel — Scene Rendering | ✅ COMPLETE | MeshTable, AssetLoader, ComponentStore→renderer wiring |
| 6 | Data Channel — PBR + Lights | ✅ COMPLETE | PipelineTable+ShaderCache, PBR/light_culling/shadow WGSL shaders |
| 7 | Command Channel — Frame Graph | ✅ COMPLETE | Full 6-phase compiler (DAG→topo→barriers→async→dead elimination) |
| 8 | Command Channel — Material DSL | ✅ COMPLETE | Python DSL+compiler, Rust DepGraph, material_compile PyO3 |
| 9 | Full Features | ✅ COMPLETE | PostProcess, Particles, DDGI — all WGSL+Rust |
| 10 | GPU Memory Management | ✅ COMPLETE | FrameAllocator, PoolAllocator, StackAllocator, GpuBudget |
| 11 | Editor Integration | ✅ COMPLETE | Editor State, REPL entity listing, 14 PyO3 functions |

## 6. Key reference documents

- [PHASE_N_TODO.md](PHASE_N_TODO.md) -- Corrected 39-task status across 11 phases
- [GAP_3_SUMMARY.md](GAP_3_SUMMARY.md) -- Deep codebase investigation with per-item verification
- [CLARIFICATION.md](CLARIFICATION.md) -- Conceptual framing, decisions, pedagogical context
- [PHASE_0_SCAFFOLDING_ARCH.md](PHASE_0_SCAFFOLDING_ARCH.md)
- [PHASE_1_TYPE_CHANNEL_ARCH.md](PHASE_1_TYPE_CHANNEL_ARCH.md)
- [PHASE_2_DATA_CHANNEL_COMPONENT_STORE_ARCH.md](PHASE_2_DATA_CHANNEL_COMPONENT_STORE_ARCH.md)
- [PHASE_3_DATA_CHANNEL_GPU_MATH_ARCH.md](PHASE_3_DATA_CHANNEL_GPU_MATH_ARCH.md)
- [PHASE_4_COMMAND_CHANNEL_TRIANGLE_ARCH.md](PHASE_4_COMMAND_CHANNEL_TRIANGLE_ARCH.md)
- [PHASE_5_DATA_CHANNEL_SCENE_RENDERING_ARCH.md](PHASE_5_DATA_CHANNEL_SCENE_RENDERING_ARCH.md)
- [PHASE_6_DATA_CHANNEL_PBR_LIGHTS_ARCH.md](PHASE_6_DATA_CHANNEL_PBR_LIGHTS_ARCH.md)
- [PHASE_7_COMMAND_CHANNEL_FRAME_GRAPH_ARCH.md](PHASE_7_COMMAND_CHANNEL_FRAME_GRAPH_ARCH.md)
- [PHASE_8_COMMAND_CHANNEL_MATERIAL_DSL_ARCH.md](PHASE_8_COMMAND_CHANNEL_MATERIAL_DSL_ARCH.md)
- [PHASE_9_FULL_FEATURES_ARCH.md](PHASE_9_FULL_FEATURES_ARCH.md)
- [PHASE_10_GPU_MEMORY_ARCH.md](PHASE_10_GPU_MEMORY_ARCH.md)
- [PHASE_11_EDITOR_ARCH.md](PHASE_11_EDITOR_ARCH.md)
