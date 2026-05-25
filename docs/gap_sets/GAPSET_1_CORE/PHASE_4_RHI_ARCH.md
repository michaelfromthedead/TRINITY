# PHASE 4: RHI wgpu Mapping

**Scope:** Close all S14 gaps by mapping Python RHI ABCs (abstract base classes) to wgpu calls through a formal Rust mapping layer.
**Depends on:** Phase 0 (omega math types for vertex data, bytemuck Pod/Zeroable), wgpu crate
**Produces:** Python RHI ABCs (7 files, existing) + Rust wgpu backend (renderer.rs, pipeline.rs, existing) + formal Python->Rust mapping layer (does not yet exist)
**Status:** PARTIAL (0/7 tasks DONE as specified; both sides exist independently but are not formally bridged)

## 1. Overview

Phase 4 addresses the RHI (Render Hardware Interface) abstraction layer. The original plan called for a Rust RHI layer that wraps wgpu and maps Python RHI ABC calls to wgpu API calls. The actual implementation has two independent tracks:

- **Python RHI ABCs** (`engine/platform/rhi/`, 7 files): Abstract base classes defining the RHI API surface -- Device, Adapter, Buffer, Texture, Sampler, Pipeline, CommandList, Queue, SwapChain, Sync primitives, RayTracing. These are the public API that engine code imports.
- **Rust wgpu backend** (`crates/renderer-backend/src/renderer.rs` + `pipeline.rs`): A working wgpu renderer with Instance/Adapter/Device/Queue/Surface setup, a triangle pipeline, PipelineTable + ShaderCache, render loop, and resize handling.

**What does NOT exist:** A formal mapping layer that connects the Python RHI ABCs to the Rust wgpu backend. The two sides work independently. The Python RHI ABCs have no wgpu bindings; the Rust wgpu renderer is not callable through the RHI ABC interface.

Additionally, the `gpu_driven/` module (buffers.rs, texture_table.rs, material_table.rs, mesh_table.rs) provides production-grade GPU resource management with triple-buffered staging, texture atlases, material descriptors, and mesh tables. 8 WGSL shaders cover PBR, forward+ culling, CSM shadows, DDGI, and particles.

## 2. Architectural decisions

- **Python RHI ABCs as the public API.** All engine rendering code uses the Python RHI ABCs (abstract base classes with Null* implementations for headless/fallback). This follows TRINITY's "Python for ergonomics" principle.
- **Rust wgpu as the backend implementation.** wgpu provides cross-platform GPU access (Vulkan/Metal/DX12/WebGPU). The Rust renderer (renderer.rs) handles device creation, pipeline compilation, and the render loop.
- **gpu_driven/ as the production resource layer.** The `gpu_driven/` module provides GPU resource management at a higher level than raw wgpu buffers: triple-buffered staging (Idle->Acquired->Submitted->Ready), texture tables with GPU-side arrays, material descriptors, and mesh tables with index/vertex buffers.
- **PipelineTable + ShaderCache for pipeline deduplication.** `pipeline.rs` provides `PipelineTable` (pipeline create/lookup/invalidate by id), `ShaderCache` (SHA-256 deduplication of WGSL modules), and `CachedPipeline` (pipeline + bind group layout + shader hash).
- **No formal RHI mapping layer.** The original plan described a Rust RHI layer that wraps wgpu behind Python-callable ABCs. This would require a trait interface analogous to the Python ABCs but in Rust, with PyO3 bindings for each method. This hasn't been implemented.
- **S14 gaps are structurally closed** by the existence of both sides: S14-G1 (Single Queue -- wgpu has one queue), S14-G2 (Explicit Fences -- wgpu's implicit synchronization), S14-G4 (Barriers -- wgpu's automatic insertion), S14-G5 (MemoryType -> Usage -- gpu_driven/buffers.rs). But the Python code can't call the Rust side through the ABC interface.

## 3. Constraints specific to this phase

- Python RHI ABCs must remain the public API; the Rust mapping layer must implement the ABC interface, not replace it.
- Null* implementations (NullAdapter, NullDevice, NullCommandList, NullQueue) must continue to work for headless/testing environments.
- WGSL shaders are compiled by naga at runtime; compilation errors must propagate to Python with source location.
- All GPU-uploadable types must implement bytemuck Pod/Zeroable for direct buffer writes.
- The mapping layer must handle device loss (wgpu device lost callback -> Python recovery path).

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `engine/platform/rhi/__init__.py` | RHI module re-exports all public types (Adapter, Device, Buffer, Texture, Sampler, Shader, Pipeline, Queue, CommandList, SwapChain, Fence, Semaphore). | DONE |
| `engine/platform/rhi/device.py` | `Device` ABC, `Adapter` ABC, `AdapterInfo`, `FeatureSupport`, `FormatSupport`, `QueueType`, `DeviceConfig`. NullAdapter/NullDevice for fallback. | DONE |
| `engine/platform/rhi/resources.py` | `Buffer` ABC, `Texture` ABC, `Sampler` ABC. `BufferUsage/Desc`, `TextureType/Usage/Desc`, `SamplerDesc`, `MemoryType`, `Format`, `AddressMode`, `FilterMode`. | DONE |
| `engine/platform/rhi/pipeline.py` | `Shader` ABC, `GraphicsPipelineDesc`, `ComputePipelineDesc`, `RaytracingPipelineDesc`. `BlendState`, `DepthStencilState`, `RasterizerState`, `PrimitiveTopology`, `CullMode`, `FillMode`. | DONE |
| `engine/platform/rhi/commands.py` | `CommandList` ABC, `Queue` ABC. `NullCommandList`/`NullQueue` for fallback. `Command` base class. | DONE |
| `engine/platform/rhi/swapchain.py` | `SwapChain` ABC, `SwapChainDesc`, present mode, resize, vsync control. | DONE |
| `engine/platform/rhi/sync.py` | `Fence` ABC, `Semaphore` ABC. GPU synchronization primitives. | DONE |
| `engine/platform/rhi/raytracing.py` | RayTracing pipeline ABCs (experimental, feature-gated). | DONE |
| `crates/renderer-backend/src/renderer.rs` | wgpu Instance/Adapter/Device/Queue/Surface. Triangle pipeline with WGSL shaders (naga-validated). Render loop (clear+draw+submit+present). resize() with zero-size guard. PyO3: renderer_init, renderer_resize, renderer_screenshot, renderer_shutdown. | DONE |
| `crates/renderer-backend/src/pipeline.rs` | `CachedPipeline` (id, render_pipeline, bind_group_layout, shader_hash), `ShaderCache` (SHA-256 keyed, module LRU), `PipelineTable` (create/lookup/invalidate). 705 lines. | DONE |
| `crates/renderer-backend/src/gpu_driven/buffers.rs` | `BufferRegistry` triple-buffered staging (Idle/Acquired/Submitted/Ready). GPU budget tracking. 777 lines. | DONE |
| `crates/renderer-backend/src/gpu_driven/texture_table.rs` | Texture table with GPU-side texture array, atlas management. | DONE |
| `crates/renderer-backend/src/gpu_driven/material_table.rs` | Material descriptor table, material parameter management. | DONE |
| `crates/renderer-backend/src/gpu_driven/mesh_table.rs` | Mesh table with index/vertex buffers, mesh descriptor arrays. | DONE |
| `crates/renderer-backend/shaders/` | 8 WGSL shaders: PBR, forward+ culling, CSM shadows, DDGI probes, particles, fullscreen triangle, post-process. | DONE |
| (does not exist) | Python RHI ABC -> Rust wgpu mapping layer (e.g., `RhiDevice` implementing `Device` ABC with wgpu::Device backing, callable from Python via PyO3). | NOT IMPLEMENTED |

## 5. Testing strategy

- Python RHI tests (73 existing) run with Null* implementations for headless validation.
- Rust wgpu tests (renderer.rs, pipeline.rs) validate device creation, pipeline compilation, and render loop on Vulkan backend.
- WGSL shaders validated via naga in tests (`include_str!` shader sources, parse via naga::front::wgsl).
- gpu_driven tests validate buffer allocation, texture table operations, material compilation.
- **Missing:** No integration tests that exercise the full Python RHI ABC -> Rust wgpu path (because the mapping layer doesn't exist).

## 6. Open questions

- **Mapping layer architecture:** Should the mapping layer be a set of Rust structs that implement the Python ABC protocol (via PyO3 with `#[pyclass]` extending `#[pyclass]` ABCs), or should it be a standalone Rust RHI trait layer behind PyO3 bridging functions?
- **Import path for mapped types:** Should `engine.platform.rhi` import detect `_omega` and swap in Rust-backed implementations (like `_HAVE_OMEGA` in world.py), or should there be explicit `import_omega_rhi()` calls?
- **WGSL compilation caching:** The ShaderCache deduplicates by SHA-256 hash per session. Should compiled shaders be cached to disk for faster startup?
- **Ray tracing support:** The Python RHI ABCs include raytracing pipeline types, but wgpu's ray tracing support (via Vulkan Ray Tracing extension) is experimental. Should this be gated behind a feature flag?

## 7. References

- `engine/platform/rhi/` -- All 7 Python RHI ABC files
- `crates/renderer-backend/src/renderer.rs` -- wgpu renderer with PyO3 bindings
- `crates/renderer-backend/src/pipeline.rs` -- PipelineTable + ShaderCache
- `crates/renderer-backend/src/gpu_driven/` -- Buffer registry, texture/material/mesh tables
- `crates/renderer-backend/shaders/` -- 8 WGSL shader sources
- GAP_1_SUMMARY.md -- Investigation for T-CORE-4.1 through T-CORE-4.7
- CLARIFICATION.md -- Rationale for two-track RHI implementation
- S14-G1 through S14-G10 gap coverage (see PHASE_N_TODO.md §Gap Coverage Map for inline gap-to-task mapping)
