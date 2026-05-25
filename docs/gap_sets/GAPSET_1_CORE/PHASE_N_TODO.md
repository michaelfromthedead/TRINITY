# GAPSET_1_CORE: Phase Tasks

## Phase 0: Deterministic Math Library (math.rs)

**Objective:** Implement all deterministic math types (vector, matrix, quaternion, fixed-point) with bytemuck Pod/Zeroable derives. ~1200 lines of Rust. Strict dependency for all subsequent phases.

### Task T-CORE-0.1: Crate Scaffolding and bytemuck Setup

- [ ] **T-CORE-0.1** Create omega crate with Cargo.toml including bytemuck, parking_lot, crossbeam, slotmap, serde, pyo3, wgpu dependencies
  - **Acceptance:** `cargo build` produces libomega.rlib with all declared dependencies resolvable
  - **Dependencies:** None
  - **Effort:** 0.5 day

### Task T-CORE-0.2: Fixed16 and Fixed32 Types

- [ ] **T-CORE-0.2** Implement Fixed16 (Q8.8, i16) and Fixed32 (Q16.16, i32) with arithmetic operators (add, sub, mul, div), comparison, conversion to/from f32, bytemuck Pod/Zeroable, serialize/deserialize
  - **Acceptance:** All arithmetic is bit-exact deterministic; Fixed16(1.5) * Fixed16(2.0) == Fixed16(3.0); no implicit float conversion; round-trip f32 conversion within 1 ULP; bytemuck::cast_slice works on slices
  - **Dependencies:** T-CORE-0.1
  - **Effort:** 1.5 days

### Task T-CORE-0.3: FVec2, FVec3, FVec4 (Fixed32 Vector Types)

- [ ] **T-CORE-0.3** Implement FVec2, FVec3, FVec4 with component access, arithmetic, dot product, cross product (FVec3 only), length, normalize, lerp, bytemuck Pod/Zeroable
  - **Acceptance:** dot(unit_x, unit_y) == Fixed32::ZERO; cross produces correct normal; length squared matches dot(self); all types derive Pod+Zeroable; bytemuck::cast_slice works
  - **Dependencies:** T-CORE-0.2
  - **Effort:** 1.5 days

### Task T-CORE-0.4: FQuat and M64 (Mat4) Types

- [ ] **T-CORE-0.4** Implement FQuat (w,x,y,z Fixed32) with identity, multiplication, conjugate, inverse, rotate_vector, slerp (threshold 0.9995), and M64 (4x4 column-major Fixed32) with identity, multiplication, inverse, transpose, look_at, perspective, bytemuck Pod/Zeroable
  - **Acceptance:** Quat identity * v == v; slerp(q, q, 0.5) == q; M64 identity * v == v; inverse(M64) * M64 == identity within epsilon; bytemuck::cast_slice produces &[[Fixed32; 16]] usable as uniform buffer; column-major layout matches WGSL mat4x4<f32> when uploaded
  - **Dependencies:** T-CORE-0.3
  - **Effort:** 2 days

### Task T-CORE-0.5: Standard Float Types (Vec2/3/4, Mat3/4, Quat) and TrigLUT

- [ ] **T-CORE-0.5a** Implement Vec2/3/4, Mat3, Mat4, Quat as standard f32 variants (non-deterministic, for rendering path) with same API surface as fixed-point variants
  - **Acceptance:** API parity with FVec/FQuat/M64; bytemuck Pod/Zeroable; Mat3*Vec3 for normal computation
  - **Effort:** 1 day

- [ ] **T-CORE-0.5b** Implement SimRng (splitmix64 deterministic PRNG) and TrigLUT (precomputed sin/cos/tan at 4096 intervals with linear interpolation)
  - **Acceptance:** SimRng produces identical sequence from same seed on x86 and ARM; TrigLUT sin/cos error < 0.0001 vs std; tests pass on both platforms
  - **Dependencies:** T-CORE-0.2
  - **Effort:** 1 day

### Task T-CORE-0.6: Math Library Tests

- [ ] **T-CORE-0.6** Write comprehensive Rust native tests organized by module: fixed-point arithmetic (100+ edge cases), vector ops (identity, zero, edge), quaternion ops (identity, slerp edge cases), matrix ops (identity, inverse edge cases), TrigLUT accuracy bounds, SimRng sequence determinism
  - **Acceptance:** All tests pass; code coverage >90% on math modules; determinism tests pass on same binary across hosts; edge cases tested (division by zero, overflow, normalization of zero vector)
  - **Dependencies:** T-CORE-0.2, T-CORE-0.3, T-CORE-0.4, T-CORE-0.5
  - **Effort:** 1 day

**Phase 0 Total: ~8.5 days**

---

## Phase 1: Memory Management + Entity System

**Objective:** Implement allocators and entity identifier. ~700 lines of Rust.

### Task T-CORE-1.1: LinearAllocator (Frame-Scoped Bump Allocator)

- [ ] **T-CORE-1.1** Implement LinearAllocator: allocate(alignment) returns aligned pointer; reset() for frame boundary; track high-water mark; DEFAULT_FRAME_ALLOCATOR_SIZE = 1MB; CACHE_LINE_BYTES (64) alignment; MemoryStats reporting
  - **Acceptance:** Allocations are 64-byte aligned; reset makes all memory available; high-water mark accurately reports peak usage; stress test of 10k allocations succeeds; concurrent reads from different threads do not false-share
  - **Dependencies:** T-CORE-0.1
  - **Effort:** 1 day

### Task T-CORE-1.2: PoolAllocator

- [ ] **T-CORE-1.2** Implement PoolAllocator: fixed-size slots pre-allocated in contiguous block; acquire() / release() O(1); free list tracking; CACHE_LINE_BYTES alignment; MemoryStats reporting
  - **Acceptance:** acquire() returns aligned slot; release() returns slot to free list; all slots acquired and released in random order with no leaks; stress test of 10k acquire/release cycles
  - **Dependencies:** T-CORE-0.1
  - **Effort:** 1 day

### Task T-CORE-1.3: RingBuffer (Staging Allocator)

- [ ] **T-CORE-1.3** Implement RingBuffer: fixed-size circular buffer with head/tail cursors; allocate(size, alignment); wrap-around on overflow; CACHE_LINE_BYTES alignment; MemoryStats reporting; overflow detection
  - **Acceptance:** Sequential allocations wrap correctly; overwrite returns error/panic in debug; alignment guaranteed; stress test of 100k allocate/wrap cycles
  - **Dependencies:** T-CORE-0.1
  - **Effort:** 1 day

### Task T-CORE-1.4: EntityId (Generational Index)

- [ ] **T-CORE-1.4** Implement EntityId: u32 wrapper with 24-bit index + 8-bit generation; EntityIdIterator for batch iteration; generation increment on despawn; null/invalid sentinel; bytemuck Pod/Zeroable; Display/Debug
  - **Acceptance:** MAX_ENTITIES = 16,777,215 (2^24-1); using stale EntityId after despawn+reuse detected via generation mismatch; EntityId(u32::MAX) is null; bytemuck::cast_slice works
  - **Dependencies:** T-CORE-0.1
  - **Effort:** 0.5 day

### Task T-CORE-1.5: Memory and Entity Tests

- [ ] **T-CORE-1.5** Write tests: allocator stress tests, alignment verification, concurrent allocation safety, EntityId generation wrap detection, EntityIdIterator correctness, memory leak detection (valgrind/drmemory CI pass)
  - **Acceptance:** All tests pass; valgrind shows no leaks; concurrent allocation from 4 threads shows no corruption
  - **Dependencies:** T-CORE-1.1, T-CORE-1.2, T-CORE-1.3, T-CORE-1.4
  - **Effort:** 0.5 day

**Phase 1 Total: ~4 days**

---

## Phase 2: Archetype ECS Runtime

**Objective:** Implement the archetype ECS with SoA storage, component type registry, command buffer, and hierarchical checksum. ~800 lines of Rust. Largest phase. Closes S15-G2, S15-G4, S15-G5, S15-G6, S15-G7.

### Task T-CORE-2.1: ComponentTypeInfo and Type Registry

- [ ] **T-CORE-2.1** Implement ComponentTypeInfo (id: u32, name: String, size_bytes: u32, fields: Vec<FieldInfo>, flags: u32); TypeRegistry (RwLock<HashMap<u32, ComponentTypeInfo>>); register_type(), lookup_type(), MAX_COMPONENTS (256) guard
  - **Acceptance:** Registering 256 types succeeds; registering 257th returns error/panic; lookup by ID returns correct type info; concurrent reads from multiple threads via RwLock; pyo3 binding receives type registration from Python
  - **Dependencies:** T-CORE-0.1 (crate), T-CORE-1.4 (EntityId)
  - **Effort:** 1.5 days

### Task T-CORE-2.2: Archetype and SoA Columns

- [ ] **T-CORE-2.2** Implement Archetype (id: u32, component_ids: Vec<u32>, columns: Vec<Vec<u8>>, entities: Vec<EntityId>, row_count: usize); column access by component ID; entity lookup by row; row removal with swap_remove (preserve density)
  - **Acceptance:** Archetype stores each component type in separate Vec<u8> column; row_count accurately reflects entity count; swap_remove maintains dense storage; adding/removing rows does not leak memory; column data is correctly aligned for component type
  - **Dependencies:** T-CORE-2.1, T-CORE-1.1, T-CORE-1.2
  - **Effort:** 2 days

### Task T-CORE-2.3: ComponentStore (archetype registry + entity operations)

- [ ] **T-CORE-2.3** Implement ComponentStore with spawn(entity_id, component_values), despawn(entity_id), read_field(entity_id, component_id, field_offset), write_field(entity_id, component_id, field_offset, value), query(component_ids) returning entity row references, column_slice(component_id) returning &[u8], all behind Arc<RwLock<ComponentStore>>
  - **Acceptance:** Spawn then read_field returns correct value; despawn marks entity for reuse; query with [Transform, Velocity] returns only entities with both; column_slice returns contiguous slice for GPU upload; concurrent reads from 4 threads are safe; write_field under RwLock write guard is exclusive
  - **Dependencies:** T-CORE-2.2, T-CORE-0.3 (FVec3), T-CORE-0.5 (Trinity component types)
  - **Effort:** 2.5 days

### Task T-CORE-2.4: CommandBuffer and Structural Changes

- [ ] **T-CORE-2.4** Implement CommandBuffer: spawn_command, despawn_command, add_component_command, remove_component_command; flush() applies all mutations to ComponentStore atomically; replay() for determinism verification; clear() for frame boundary
  - **Acceptance:** Commands collected during frame execution are applied in order at flush(); replay produces identical state; clear() empties buffer without applying; concurrent command recording from worker threads is safe
  - **Dependencies:** T-CORE-2.3
  - **Effort:** 1.5 days

### Task T-CORE-2.5: HierarchicalChecksum and System Phases

- [ ] **T-CORE-2.5a** Implement HierarchicalChecksum: per-entity rolling checksum of component data; world-level checksum from entity checksums; frame boundary verification endpoint; xxhash-like fast hash
  - **Acceptance:** Checksum changes when any entity field changes; same state produces same checksum; collision resistance sufficient for determinism verification (not cryptographic)
  - **Effort:** 1 day

- [ ] **T-CORE-2.5b** Implement SystemPhase (ordered collection of systems) and SystemContext (delta_time: Fixed32, command_buffer: &mut CommandBuffer, world_checksum: &HierarchicalChecksum); system trait with run(context) method
  - **Acceptance:** Systems execute in declared order; SystemContext provides correct delta_time; systems can read/write components via context; checksum available for verification
  - **Dependencies:** T-CORE-2.4
  - **Effort:** 1 day

### Task T-CORE-2.6: ECS Tests

- [ ] **T-CORE-2.6** Write tests: archetype move (add/remove component changes archetype), 10k entity spawn/despawn stress, concurrent query from 4 threads, command buffer flush correctness, checksum determinism across identical workloads, system phase execution ordering, column_slice GPU readiness
  - **Acceptance:** All tests pass; 10k entity stress completes under 100ms; concurrent queries return consistent results; command buffer replay produces identical state; checksum is deterministic across runs
  - **Dependencies:** T-CORE-2.3, T-CORE-2.4, T-CORE-2.5
  - **Effort:** 1 day

**Phase 2 Total: ~10.5 days**

---

## Phase 3: Task/Job System

**Objective:** Implement work-stealing thread pool with job graph dependencies. ~600 lines of Rust. Closes S15-G3, S15-G8.

### Task T-CORE-3.1: ThreadPool with Work-Stealing

- [ ] **T-CORE-3.1** Implement ThreadPool: spawn N worker threads; each thread has local deque; stolen tasks from sibling deques; DEFAULT_WORKER_COUNT = 0 means auto-detect; shutdown joins all threads gracefully; priority-aware scheduling (6 levels)
  - **Acceptance:** ThreadPool with 4 workers runs 4 tasks concurrently; work-stealing balances load across workers; shutdown completes all submitted tasks before returning; priority HIGH tasks execute before LOW tasks
  - **Dependencies:** T-CORE-0.1 (crate), crossbeam
  - **Effort:** 1.5 days

### Task T-CORE-3.2: JobGraph and Dependencies

- [ ] **T-CORE-3.2** Implement JobGraph (compiled dependency DAG), JobGraphBuilder (add_task, depends_on, finalize), TaskHandle (non-blocking is_complete), execution engine (submit job graph, workers execute tasks that have all dependencies met)
  - **Acceptance:** Task A(->B) means B starts only after A completes; diamond dependency (A->B, A->C, B->C) works correctly; cycle detection on build returns error; 1000-node graph builds and executes; TaskHandle.is_complete() returns correct status
  - **Dependencies:** T-CORE-3.1
  - **Effort:** 1.5 days

### Task T-CORE-3.3: parallel_for

- [ ] **T-CORE-3.3** Implement parallel_for(range, chunk_size, function) splitting range into chunks distributed via job graph; configurable chunk size (adaptive: auto-sized to worker count); returns when all chunks complete
  - **Acceptance:** parallel_for(0, 1000, f) calls f for each index 0..999 exactly once; chunks are distributed across all workers; chunk_size=0 auto-sizes to num_workers; speedup vs sequential >3x on 4-core
  - **Dependencies:** T-CORE-3.2
  - **Effort:** 1 day

### Task T-CORE-3.4: Task System Tests

- [ ] **T-CORE-3.4** Write tests: thread pool work-stealing verification, job graph dependency correctness, cycle detection, 10k task throughput, parallel_for index coverage, priority inversion test (high priority tasks always complete before low priority), single-threaded mode (0 workers) determinism
  - **Acceptance:** All tests pass; 10k task throughput >50k tasks/sec on 4-core; single-threaded mode produces deterministic execution order; priority inversion is impossible by design
  - **Dependencies:** T-CORE-3.1, T-CORE-3.2, T-CORE-3.3
  - **Effort:** 0.5 day

**Phase 3 Total: ~4.5 days**

---

## Phase 4: RHI wgpu Mapping

**Objective:** Close all 10 S14 gaps by mapping Python RHI ABCs to wgpu. ~7 work packages. Closes S14-G1 through S14-G10 and cross-cutting gaps 3.6, 3.12.

### Task T-CORE-4.1: Device/Adapter Layer (S14-A)

- [ ] **T-CORE-4.1** Implement Rust RHI device: Instance creation, adapter request (AdapterSelector: low-power/high-performance/none), device request (FeatureFlags, QualityTiers), WgpuFence wrapper (submission index counter, wait() polls completed_index), single Queue mapped from all three Python queue types, device lost callback handling (S14-G1, S14-G2, S14-G8, S14-G9, S14-G10)
  - **Acceptance:** Adapter selection respects power preference; device creation with feature flags succeeds; all 3 Python queue types map to same wgpu::Queue; WgpuFence.wait() returns when submission completes; device lost triggers resource re-creation path
  - **Dependencies:** T-CORE-0.1 (crate), wgpu
  - **Effort:** 3 days

### Task T-CORE-4.2: Buffer/Texture/Sampler Layer (S14-B)

- [ ] **T-CORE-4.2** Implement Rust RHI resources: Buffer creation (WgpuBuffer mapping Python MemoryType to Usage flags), Texture creation (WgpuTexture mapping TextureType to Extent3D, support CUBE/ARRAY), Sampler creation, MemoryType->wgpu Usage translation table (S14-G5, S14-G6)
  - **Acceptance:** Buffer with MemoryType.UPLOAD maps to COPY_SRC | MAP_WRITE; Texture with CUBE creates Extent3D with depth=6; sampler with linear filtering creates wgpu::SamplerDescriptor correctly
  - **Dependencies:** T-CORE-4.1
  - **Effort:** 3 days

### Task T-CORE-4.3: Pipeline Layer (S14-C)

- [ ] **T-CORE-4.3** Implement Rust RHI pipelines: RenderPipeline creation (vertex layout, fragment shader, blend state, depth/stencil), ComputePipeline creation, PipelineLayout with BindGroupLayout, ShaderModule from WGSL source, compilation error reporting
  - **Acceptance:** Pipeline creation with vertex+fragment shaders succeeds; WGSL compilation errors are reported with source location; compute pipeline with workgroup size creates correctly; pipeline layout matches shader reflection
  - **Dependencies:** T-CORE-4.2
  - **Effort:** 4 days

### Task T-CORE-4.4: Command Recording Layer (S14-D)

- [ ] **T-CORE-4.4** Implement Rust RHI command recording: CommandEncoder wrapping wgpu::CommandEncoder, RenderPass (begin/end, draw, set_pipeline, set_bind_group, set_vertex_buffer, set_index_buffer), ComputePass (begin/end, dispatch), barrier() as no-op (S14-G4)
  - **Acceptance:** RenderPass records draw commands correctly; barrier() compiles to no-op; command buffer submit succeeds on device; encoding error produces clear message
  - **Dependencies:** T-CORE-4.3
  - **Effort:** 2 days

### Task T-CORE-4.5: Swapchain Layer (S14-E)

- [ ] **T-CORE-4.5** Implement Rust RHI swapchain: Surface creation from window handle, SwapChain configuration (format, size, present_mode), get_current_texture(), present(), resize handling, vsync/immediate mode
  - **Acceptance:** Swapchain configures with window size; get_current_texture returns valid texture view; present() presents to screen; resize reconfigures swapchain; vsync and immediate modes work
  - **Dependencies:** T-CORE-4.1
  - **Effort:** 2 days

### Task T-CORE-4.6: Bind Group Layer (S14-F)

- [ ] **T-CORE-4.6** Implement Rust RHI bind groups: WgpuBindGroupManager (cache bind groups by layout+resource hash), BindGroupLayout creation, BindGroup creation from descriptor set, cache eviction policy (frame-scoped), deferred binding for dynamic resources (S14-G3)
  - **Acceptance:** Bind group creation from same layout+resources returns cached entry; cache eviction happens at frame boundary; bind group layout matches shader expectations; dynamic uniform buffers work
  - **Dependencies:** T-CORE-4.2
  - **Effort:** 2 days

### Task T-CORE-4.7: RHI Integration Tests (S14-G)

- [ ] **T-CORE-4.7** Write Rust-native RHI integration tests: device creation, buffer upload/readback, texture creation and sampling, render pipeline draw to texture, compute pipeline dispatch, swapchain acquire+present cycle, bind group caching, multi-frame stress test, device lost recovery. Re-run all 73 existing Python RHI tests against Rust implementation.
  - **Acceptance:** All Rust RHI tests pass on Vulkan backend; all 73 Python RHI tests pass against Rust backing (via bridge in Phase 5); multi-frame stress (1000 frames) shows no leaks; no test depends on GPU availability (mock/headless fallback)
  - **Dependencies:** T-CORE-4.1 through T-CORE-4.6, T-CORE-5.2 (bridge for Python test pass)
  - **Effort:** 3 days

**Phase 4 Total: ~19 days**

---

## Phase 5: Python-Side Bridge Wiring

**Objective:** Wire Python layer to Rust via 3-channel Bridge. Closes S15-G1, S15-G9, S15-G10 and cross-cutting gaps 3.1, 3.2.

### Task T-CORE-5.1: PyO3 Bridge Foundation

- [ ] **T-CORE-5.1** Create PyO3 bridge module in omega crate: define Python-facing functions (_omega::component_read, component_write, type_register, world_create, world_spawn, world_despawn, world_query, scheduler_step, scheduler_submit_job_graph); error handling (PyErr for validation failures); type conversion (Python float->f32/int->i32/bool->u8/str->string); GIL release for long-running Rust calls
  - **Acceptance:** All bridge functions callable from Python; type conversion round-trips correctly; validation errors raise Python exceptions; GIL is released during ECS query operations
  - **Dependencies:** T-CORE-0.1 (crate with pyo3)
  - **Effort:** 2 days

### Task T-CORE-5.2: Bridge 3-Channel Protocol

- [ ] **T-CORE-5.2** Implement 3-channel protocol: Type channel (type_register called at Python import/definition time), Data channel (component_read/write called per-frame per-field), Command channel (spawn/despawn via CommandBuffer bridge); each channel has distinct Python entry point; validation layer on Type channel catches layout mismatches; Data channel is the hot path (optimized for minimal overhead)
  - **Acceptance:** Type registration at import time populates Rust TypeRegistry; Data channel read/write round-trips in <100ns per field (benchmarked); Command channel spawn+flush creates entity in Rust store; validation rejects mismatched types at definition time
  - **Dependencies:** T-CORE-5.1, T-CORE-2.1 (TypeRegistry), T-CORE-2.4 (CommandBuffer)
  - **Effort:** 2 days

### Task T-CORE-5.3: RustStorageDescriptor and ComponentMeta Step 6b

- [ ] **T-CORE-5.3a** Implement RustStorageDescriptor in Python: _get_stored() calls _omega::component_read(), _set_stored() calls _omega::component_write(); assigned to component class __dict__ replacement (not __slots__); fallback to Python StorageDescriptor for non-component objects
  - **Acceptance:** Reading component_instance.position from Python returns value from Rust SoA; writing component_instance.position = x writes to Rust SoA; non-component objects continue using Python StorageDescriptor
  - **Effort:** 1.5 days

- [ ] **T-CORE-5.3b** Modify ComponentMeta.__new__() step 6b: after Python class is defined, call _omega::type_register() with computed layout; _build_rust_layout() maps Python types: float->f32/f64, int->i32/u32/i64/u64, bool->u8, str->String; field name, type code, offset, total size published to Rust
  - **Acceptance:** Defining class Position(Component): x: float, y: float automatically registers type with id, 8-byte size, two f32 fields at offsets 0 and 4; duplicate definition returns same type id; type registration is idempotent
  - **Dependencies:** T-CORE-5.2 (Type channel)
  - **Effort:** 1.5 days

### Task T-CORE-5.4: World and Entity Python Classes

- [ ] **T-CORE-5.4** Create/modify trinity/world.py: World class with spawn(components: dict) -> EntityId, despawn(entity_id), query(*component_types) -> iterator of entity tuples, create() / destroy() world lifecycle; Entity class with get_component(type) -> component instance, set_component(component), has_component(type) -> bool; all methods delegate to _omega bridge calls
  - **Acceptance:** world.spawn(Position(1,2)) creates entity in Rust; world.query(Position, Velocity) returns correct tuples; entity.get_component(Position) returns correct values; entity.set_component(position) writes to Rust; world lifecycle creates/destroys Rust store
  - **Dependencies:** T-CORE-5.2, T-CORE-2.3 (ComponentStore), T-CORE-2.4 (CommandBuffer)
  - **Effort:** 2 days

### Task T-CORE-5.5: Scheduler Bridge and Frame Loop

- [ ] **T-CORE-5.5** Modify trinity/omega/scheduler.py: frame loop dispatches system phases as job graphs to Rust ThreadPool via bridge; each phase transition triggers CommandBuffer flush; frame start allocates LinearAllocator reset; frame end submits HierarchicalChecksum verification; Python systems execute in phase order alongside Rust systems
  - **Acceptance:** Frame loop executes Phase 0 systems before Phase 1 systems; CommandBuffer flush happens at phase boundaries; HierarchicalChecksum matches between Python and Rust views of world state; scheduler can run in single-threaded debug mode
  - **Dependencies:** T-CORE-5.4, T-CORE-3.2 (JobGraph), T-CORE-2.5 (CommandBuffer + Checksum)
  - **Effort:** 1.5 days

### Task T-CORE-5.6: Bridge Integration Tests

- [ ] **T-CORE-5.6** Write bridge integration tests: Python import triggers type registration (verify in Rust); create entity from Python, read field from Python, modify from Rust side, verify in Python; 3-channel protocol stress test (10k type registrations, 1M field reads, 10k spawn/despawn); determinism test (identical Python script produces identical world checksum on rerun); GIL release test (Python thread creates entities while another queries)
  - **Acceptance:** All integration tests pass; type registration round-trip verified; 1M field reads complete under 100ms; world checksum is deterministic across reruns; GIL release allows concurrent access
  - **Dependencies:** T-CORE-5.2, T-CORE-5.3, T-CORE-5.4, T-CORE-5.5
  - **Effort:** 1 day

**Phase 5 Total: ~11.5 days**

---

## Summary

| Phase | Tasks | Effort (days) | Rust Lines | Python Lines | Gaps Closed |
|-------|-------|---------------|------------|--------------|-------------|
| Phase 0: Math Library | 6 | 8.5 | ~1200 | 0 | 3.4, S15-G1 (partial) |
| Phase 1: Memory + Entity | 5 | 4 | ~700 | 0 | S15-G6 (partial) |
| Phase 2: ECS Runtime | 6 | 10.5 | ~800 | 0 | S15-G2, S15-G4, S15-G5, S15-G6, S15-G7 |
| Phase 3: Task System | 4 | 4.5 | ~600 | 0 | S15-G3, S15-G8 |
| Phase 4: RHI wgpu | 7 | 19 | ~800 | 0 | S14-G1..G10, 3.6, 3.12 |
| Phase 5: Python Wiring | 6 | 11.5 | ~300 | ~800 | 3.1, 3.2, S15-G9, S15-G10 |
| **Total** | **34** | **58** | **~4400** | **~800** | **20 + 5 cross-cutting** |

**Total estimated effort: ~58 developer-days (11.6 weeks for single developer, 4-6 weeks with team of 2-3)**

## Phase-Level Task Dependency Summary

| Task | Depends On | Required By |
|------|-----------|-------------|
| T-CORE-0.1 | None | All |
| T-CORE-0.2 | T-CORE-0.1 | T-CORE-0.3, T-CORE-0.5b |
| T-CORE-0.3 | T-CORE-0.2 | T-CORE-0.4, T-CORE-2.3 |
| T-CORE-0.4 | T-CORE-0.3 | T-CORE-2.5b (delta_time type) |
| T-CORE-0.5a | T-CORE-0.1 | T-CORE-4.3 (vertex buffers) |
| T-CORE-0.5b | T-CORE-0.2 | T-CORE-5.5 (sim determinism) |
| T-CORE-1.1 | T-CORE-0.1 | T-CORE-2.2, T-CORE-5.5 |
| T-CORE-1.2 | T-CORE-0.1 | T-CORE-2.2 |
| T-CORE-1.3 | T-CORE-0.1 | T-CORE-4.2 (staging uploads) |
| T-CORE-1.4 | T-CORE-0.1 | T-CORE-2.1 |
| T-CORE-2.1 | T-CORE-1.4 | T-CORE-2.2, T-CORE-5.2 |
| T-CORE-2.2 | T-CORE-2.1, T-CORE-1.1, T-CORE-1.2 | T-CORE-2.3 |
| T-CORE-2.3 | T-CORE-2.2, T-CORE-0.3 | T-CORE-2.4, T-CORE-5.4 |
| T-CORE-2.4 | T-CORE-2.3 | T-CORE-2.5b, T-CORE-5.2, T-CORE-5.5 |
| T-CORE-2.5a | T-CORE-2.3 | T-CORE-2.5b |
| T-CORE-2.5b | T-CORE-2.4, T-CORE-2.5a | T-CORE-5.5 |
| T-CORE-3.1 | T-CORE-0.1, crossbeam | T-CORE-3.2 |
| T-CORE-3.2 | T-CORE-3.1 | T-CORE-3.3, T-CORE-5.5 |
| T-CORE-3.3 | T-CORE-3.2 | T-CORE-5.5 (optional) |
| T-CORE-4.1 | wgpu | T-CORE-4.2 |
| T-CORE-4.2 | T-CORE-4.1 | T-CORE-4.3, T-CORE-4.6 |
| T-CORE-4.3 | T-CORE-4.2 | T-CORE-4.4 |
| T-CORE-4.4 | T-CORE-4.3 | Standalone |
| T-CORE-4.5 | T-CORE-4.1 | Standalone |
| T-CORE-4.6 | T-CORE-4.2 | Standalone |
| T-CORE-4.7 | T-CORE-4.1..4.6, T-CORE-5.2 | -- (final verification) |
| T-CORE-5.1 | T-CORE-0.1, pyo3 | T-CORE-5.2 |
| T-CORE-5.2 | T-CORE-5.1, T-CORE-2.1, T-CORE-2.4 | T-CORE-5.3, T-CORE-5.4, T-CORE-4.7 |
| T-CORE-5.3a | T-CORE-5.2 | Standalone |
| T-CORE-5.3b | T-CORE-5.2 | Standalone |
| T-CORE-5.4 | T-CORE-5.2, T-CORE-2.3, T-CORE-2.4 | T-CORE-5.5 |
| T-CORE-5.5 | T-CORE-5.4, T-CORE-3.2, T-CORE-2.5 | Standalone |
| T-CORE-5.6 | T-CORE-5.2, T-CORE-5.3, T-CORE-5.4, T-CORE-5.5 | -- (final verification) |

## TRUE_GAPS.md Coverage Map

| Gap ID | Description | Closed By |
|--------|-------------|-----------|
| S14-G1 | Single Queue vs Multiple Queue Types | T-CORE-4.1 |
| S14-G2 | Explicit Fences vs Implicit Polling | T-CORE-4.1 |
| S14-G3 | Descriptor Heap vs Bind Group | T-CORE-4.6 |
| S14-G4 | Explicit Barriers vs Automatic Insertion | T-CORE-4.4 |
| S14-G5 | MemoryType enum vs wgpu Usage Flags | T-CORE-4.2 |
| S14-G6 | TextureType.CUBE/ARRAY vs Extent3D | T-CORE-4.2 |
| S14-G7 | Ray Tracing Pipeline experimental | T-CORE-4.1 (feature gate) |
| S14-G8 | Device lost handling | T-CORE-4.1 |
| S14-G9 | Synchronization primitives | T-CORE-4.1 |
| S14-G10 | Feature query mapping | T-CORE-4.1 |
| S15-G1 | No GPU math library | T-CORE-0.2..0.5 |
| S15-G2 | No ECS runtime | T-CORE-2.1..2.3 |
| S15-G3 | No task system | T-CORE-3.1..3.3 |
| S15-G4 | No determinism system | T-CORE-2.5, T-CORE-0.5b |
| S15-G5 | No component storage | T-CORE-2.2, T-CORE-2.3 |
| S15-G6 | No memory management | T-CORE-1.1..1.3 |
| S15-G7 | No entity system | T-CORE-1.4, T-CORE-2.3 |
| S15-G8 | No parallel execution | T-CORE-3.1..3.3 |
| S15-G9 | No bridge protocol | T-CORE-5.1, T-CORE-5.2 |
| S15-G10 | No Python-Rust interop | T-CORE-5.1, T-CORE-5.3 |
| 3.1 | Bridge Channel Protocol (CRITICAL) | T-CORE-5.2 |
| 3.2 | Core ECS (CRITICAL) | T-CORE-2.1..2.5, T-CORE-5.4 |
| 3.3 | No GPU Implementation (CRITICAL) | T-CORE-4.1..4.7 |
| 3.4 | No GPU Math Library (CRITICAL) | T-CORE-0.2..0.5 |
| 3.12 | GPU Memory Management (HIGH) | T-CORE-4.2, T-CORE-1.3 |
