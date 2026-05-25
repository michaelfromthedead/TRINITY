# PHASE 7: Command Channel -- Frame Graph

**Scope:** Execute a compiled frame graph in Rust -- DAG-sort passes, allocate/alias transient resources, insert GPU barriers, submit command buffers to wgpu.
**Depends on:** Phase 4 (wgpu device/queue for actual command execution)
**Produces:** FrameGraphExecutor with full compiler pipeline (DAG building, resource aliasing, barrier scheduling, async scheduling, dead pass elimination)
**Status:** MOST COMPLETE -- The Rust frame graph IR at `crates/renderer-backend/src/frame_graph/mod.rs` (1,681 lines) defines the full type system: resource handles, pass types (Graphics/Compute/Copy/RayTracing), edges with RAW/WAR/WAW classification, attachment descriptions, and view types. 25+ unit tests pass. The Python frame graph compiler (`engine/rendering/framegraph/`) produces the same IR concepts. The compiler phases (DAG building through dead pass elimination) are documented in code comments as phases 1-6; phase 1 (IR type definitions) is complete, phases 2-6 are stubbed.

## 1. Overview

The Frame Graph is the scheduling backbone of the GPU renderer. A frame graph is a directed acyclic graph (DAG) where nodes are render passes and edges represent resource dependencies. The Rust IR defines the complete type vocabulary: `IrPass` (graphics/compute/copy/ray_tracing variants), `IrResource` (texture2D, texture3D, buffer), `IrEdge` (RAW/WAR/WAW), and supporting types for attachments, load/store ops, view types, and instance/dispatch sources. The compiler transforms a high-level pass declaration into an optimized execution schedule with minimal barriers and maximal resource aliasing.

## 2. Architectural decisions

- **Six-phase compilation pipeline** (documented in frame_graph/mod.rs comments):
  1. **IR Construction** (COMPLETE) -- Parse pass declarations into IrPass/IrResource/IrEdge objects
  2. **DAG Builder** (STUBBED) -- Topological sort, detect cycles, compute pass ordering
  3. **Resource Aliasing** (STUBBED) -- Overlap transient resources that are never simultaneously live
  4. **Barrier Scheduling** (STUBBED) -- Insert wgpu `Barrier` objects at RAW/WAR/WAW edges
  5. **Async Scheduling** (STUBBED) -- Overlap compute and copy passes on separate queues
  6. **Dead Pass Elimination** (STUBBED) -- Remove passes whose outputs are never consumed
- **Python compiler is complementary, not redundant**: `engine/rendering/framegraph/` implements the high-level graph definition API and resource manager. Python handles the ergonomic layer; Rust handles the optimization and execution layer. The missing link is serialization (Python->Rust IR).
- **PassType enum drives pipeline creation**: Graphics, Compute, Copy, and RayTracing variants directly map to wgpu pipeline types. The variant carries type-specific fields (color attachments, compute workgroup size, copy source/destination).
- **ResourceAccessSet tracks per-pass access patterns**: Each pass declares which resources it reads (SAMPLED, UNIFORM) and writes (RENDER_TARGET, STORAGE). The barrier scheduler uses this to determine required texture state transitions.

## 3. Constraints specific to this phase

- Resource aliasing requires transient resource lifetimes to be known before barrier scheduling. The DAG builder must compute liveness intervals first.
- Barrier insertion must respect wgpu's texture state machine (Undefined -> RenderAttachment -> ShaderReadOnly -> etc.) -- wrong transitions cause validation errors or GPU hangs.
- Async scheduling requires wgpu queues to support the `COPY` and `COMPUTE` queue types, which is device-dependent.
- RayTracing passes require the `wgpu::Features::RAY_TRACING` feature flag, which is not available on all hardware.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `frame_graph/mod.rs` (1,681 lines) | IR type system + 25+ unit tests | EXISTS -- phase 1 complete |
| `IrPass` with `PassType` enum | Graphics/Compute/Copy/RayTracing variants | EXISTS |
| `IrResource` with descriptors | Texture/Buffer resource definitions | EXISTS |
| `IrEdge` with RAW/WAR/WAW | Edge classification for dependency tracking | EXISTS |
| `ResourceAccess/ResourceState` | Per-pass access patterns | EXISTS |
| `AttachmentLoadOp/AttachmentStoreOp` | Load/store semantics (Clear/Load/Store) | EXISTS |
| `ViewType` (9 variants) | Texture view types | EXISTS |
| DAG Builder | Topological sort implementation | STUBBED (phase 2) |
| Resource Aliasing | Transient resource overlap | STUBBED (phase 3) |
| Barrier Scheduling | wgpu barrier insertion | STUBBED (phase 4) |
| Async Scheduling | Queue overlap optimization | STUBBED (phase 5) |
| Dead Pass Elimination | Unreferenced output pruning | STUBBED (phase 6) |
| `engine/rendering/framegraph/frame_graph.py` | Python frame graph API | EXISTS |
| `engine/rendering/framegraph/pass_node.py` | Python pass definitions | EXISTS |
| `engine/rendering/framegraph/barrier_manager.py` | Python barrier tracking | EXISTS |
| `engine/rendering/framegraph/async_scheduler.py` | Python async scheduling | EXISTS |
| `engine/rendering/framegraph/resource_manager.py` | Python resource management | EXISTS |
| Python-to-Rust serialization | Serialize compiled graph for Rust | NOT IMPLEMENTED |

## 5. Testing strategy

- Unit (Rust): 25+ existing tests cover handle types, pass creation, resource descriptors, edge construction, round-trip serialization.
- Unit: Add tests for DAG builder (topological sort of known graph shapes -- linear, diamond, disconnected).
- Unit: Add tests for resource aliasing (two passes with non-overlapping lifetimes should share memory).
- Integration: Build a simple two-pass graph (background clear + triangle), compile through all 6 phases, verify barrier count and ordering.

## 6. Open questions

- Should the Python frame graph compiler serialize to a protocol buffer/JSON/msgpack for Rust consumption? JSON is simplest but verbose; msgpack is compact but adds a dependency. Protocol buffers add a codegen step.
- The IR supports RayTracing passes, but no hardware with DXR/Vulkan ray tracing is currently targeted. Should RayTracing remain in the type system (future-proofing) or be deferred?

## 7. References

- Phase 4 (wgpu Renderer) is the executor that runs compiled frame graphs.
- Phase 6 (PBR + Lights) creates passes that the frame graph schedules.
- Phase 9 (Full Features) adds post-process/particles/DDGI passes to the graph.
- GAP_3_SUMMARY.md section "Phase 7: Frame Graph -- MOST COMPLETE" (detailed analysis of the 1,681-line IR).
