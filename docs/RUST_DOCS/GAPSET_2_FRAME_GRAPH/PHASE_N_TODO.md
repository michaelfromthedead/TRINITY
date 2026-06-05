# GAPSET_2_FRAME_GRAPH -- Tasks

> **TASK_ID format:** T-FG-{PHASE}.{N}
> **Coverage:** All 8 S1 gaps (S1-G1 through S1-G8)
> **Effort:** L = <1 day, M = 1-3 days, H = 3-7 days, XH = 7-14 days

---

## Phase 1: Compiler Foundation + IR (Covers S1-G1, S1-G2)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-1.1 | Define `IrPass`, `IrResource`, `IrEdge` Rust structs with full field coverage (pass name, type, resource access sets, color attachments, depth stencil, instance source, dispatch source, view type) | S1-G1 | M | None |
| T-FG-1.2 | Define `PyPassNode` → `IrPass` conversion via `From<PyPassNode> for IrPass` with validation (null checks, type field ranges, format compatibility) | S1-G1 | M | T-FG-1.1 |
| T-FG-1.3 | Define `PyResourceDesc` → `IrResource` conversion with format validation, usage flag coalescing, and handle resolution | S1-G1 | M | T-FG-1.1 |
| T-FG-1.4 | Define `View` trait in Rust with `fn bind(&self, ctx: &RenderContext) -> Vec<wgpu::BindGroup>` and `Send + Sync` bounds. Provide `EmptyView` and `CameraView` implementations | S1-G2 | M | None |
| T-FG-1.5 | Add `view: Box<dyn View>` field to every `CompiledPass` variant (Graphics, Compute, Copy, RayTracing). Copy passes get `EmptyView` | S1-G2 | L | T-FG-1.4 |
| T-FG-1.6 | Implement Python-side `Registry` snapshot: `FrameGraph.compile()` collects all registered passes from the Python pass registry, serializes to `Vec<PyPassNode>`, and calls the Rust compiler via PyO3 | S1-G1 | H | T-FG-1.2, T-FG-1.3 |
| T-FG-1.7 | Implement `MockPassNode` and `MockResourceDesc` constructors for Rust-side compiler testing without Python | S1-G1 | M | T-FG-1.1 |
| T-FG-1.8 | Acceptance: 10 passes compile-correctly in Rust-side test with varying dependency patterns. Pass nodes serialize and deserialize across FFI without data loss | S1-G1 | -- | All above |

---

## Phase 2: Dependency DAG (Covers S1-G4)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-2.1 | Implement DAG builder: iterate all passes, for each pass's read set find the producing (writing) pass, add RAW edge. For each pass's write set find other passes writing the same resource, add WAW edge | S1-G4 | M | T-FG-1.1 |
| T-FG-2.2 | Implement Kahn's algorithm topological sort with in-degree tracking and FIFO processing queue | S1-G4 | M | T-FG-2.1 |
| T-FG-2.3 | Implement cycle detection: when Kahn's algorithm terminates with unprocessed nodes, run DFS from the first unprocessed node to find the cycle path. Emit resource-level diagnostic: "Cycle: Pass A writes R → Pass B reads R → Pass C writes R" | S1-G4 | H | T-FG-2.2 |
| T-FG-2.4 | Assign topological depth to each pass (longest path from any entry node with in-degree 0) | S1-G4 | L | T-FG-2.2 |
| T-FG-2.5 | Identify connected components and parallel regions: passes at the same depth with no path between them are in a parallel region. Output as `Vec<Vec<usize>>` (region index → pass indices) | S1-G4 | M | T-FG-2.4 |
| T-FG-2.6 | Unit tests: linear chain DAG, diamond DAG, multi-edge DAG, DAG with RAW+WAW edges, deliberately cyclic DAG (expect error), single-pass DAG | S1-G4 | M | T-FG-2.2, T-FG-2.3 |
| T-FG-2.7 | Acceptance: 10-pass DAG with 20+ edges compiles in <1ms. Cycle errors include resource-level diagnostic. All 6 test scenarios pass | S1-G4 | -- | All above |

---

## Phase 3: Resource Aliasing (Covers S1-G5)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-3.1 | Implement lifetime interval computation: for each transient resource, `first_pass = min(pass_index for passes that write or read R)`. `last_pass = max(pass_index for passes that write or read R)`. Interval = `[first_pass, last_pass]` | S1-G5 | M | T-FG-2.2 |
| T-FG-3.2 | Implement interference graph builder: two resources interfere iff their intervals overlap OR their formats are incompatible. Output as adjacency list `HashMap<ResourceHandle, Vec<ResourceHandle>>` | S1-G5 | M | T-FG-3.1 |
| T-FG-3.3 | Implement greedy graph coloring (largest-first): sort resources by interval length descending. For each, assign first color not used by any already-colored interfering resource | S1-G5 | M | T-FG-3.2 |
| T-FG-3.4 | Implement `ResourceAllocator` that creates wgpu textures and buffers per color group. Uses texture array layers for 2D resources, buffer offsets for buffer resources. Validates format + dimension compatibility within each group | S1-G5 | H | T-FG-3.3 |
| T-FG-3.5 | Implement history resource ring buffer: allocate `history_length` slots per history resource. Frame N uses slot `N % history_length`. Wire up slot resolution in the compiler | S1-G5 | M | T-FG-2.2 |
| T-FG-3.6 | Implement external resource import: accept opaque handles and initial `ResourceState`. No allocation -- just state tracking | S1-G5 | L | None |
| T-FG-3.7 | Build allocation table: `HashMap<ResourceHandle, (wgpu::Texture, u32 layer_or_offset)>` mapping logical handles to physical resources | S1-G5 | M | T-FG-3.4 |
| T-FG-3.8 | Unit tests: non-overlapping intervals (expect aliasing), overlapping intervals (no aliasing), format-incompatible aliasing (no aliasing), single-resource, all-resources-same-interval | S1-G5 | M | T-FG-3.1, T-FG-3.2, T-FG-3.3 |
| T-FG-3.9 | Acceptance: 15-transient-resource standard frame achieves 40%+ memory savings over independent allocation. History resources persist correctly across N frames. External resources import with correct initial state | S1-G5 | -- | All above |

---

## Phase 4: Barrier Insertion (Covers S1-G6)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-4.1 | Implement `ResourceStateTracker`: `HashMap<ResourceHandle, ResourceState>` tracking current state per resource. Method `transition(resource, from_state, to_state) -> Option<Barrier>` | S1-G6 | M | None |
| T-FG-4.2 | Implement barrier record generation: walk passes in execution order. For each resource accessed by the pass, compute required state. If state transition needed, emit `Barrier { resource, from_state, to_state, pipeline_stage_from, pipeline_stage_to }` | S1-G6 | M | T-FG-4.1 |
| T-FG-4.3 | Implement barrier batching: accumulate pending barriers during pass traversal. Before each pass `begin()`, flush all pending barriers as a single `BarrierBatch`. Clear batch after flush | S1-G6 | M | T-FG-4.2 |
| T-FG-4.4 | Implement redundant barrier elimination: before flushing a batch, scan for adjacent A→B→A transitions on the same resource. Remove both. Run a second pass for B→A→B patterns | S1-G6 | M | T-FG-4.3 |
| T-FG-4.5 | Implement wgpu barrier command generation: translate `Barrier` records to `wgpu::CommandEncoder::texture_barrier()` or `buffer_barrier()` calls. Map TRINITY `ResourceState` to wgpu `TextureUsage` / `BufferUsage` per the texture state mapping table (S1_FRAME_GRAPH.md Section 6.2) | S1-G6 | H | T-FG-4.2 |
| T-FG-4.6 | Implement per-pass `pre_barriers: Vec<Barrier>` and `post_barriers: Vec<Barrier>` on `ScheduledPass` struct, populated by the barrier scheduler | S1-G6 | M | T-FG-4.3 |
| T-FG-4.7 | Unit tests: all valid state transitions (state machine table coverage), UNINITIALIZED first-use, same-state no-op, A→B→A redundant elimination, per-pass boundary batching | S1-G6 | H | T-FG-4.1, T-FG-4.2, T-FG-4.4 |
| T-FG-4.8 | Acceptance: Correct barriers for all 10 resource states. No redundant barriers. Batching produces fewer wgpu::Command calls than per-resource naive approach. All 72 state transition pairs tested | S1-G6 | -- | All above |

---

## Phase 5: Async Compute Scheduling (Covers S1-G7)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-5.1 | Implement async candidate identification: for each compute pass, check RAW hazards (reads resource written by active graphics pass), WAW hazards (writes resource also written by active graphics pass). Pass must have neither | S1-G7 | M | T-FG-2.2 |
| T-FG-5.2 | Implement secondary timeline builder: group async-eligible compute passes into a `Vec<ScheduledPass>` ordered by their internal dependencies. Non-eligible compute passes remain on the graphics timeline | S1-G7 | M | T-FG-5.1 |
| T-FG-5.3 | Implement sync point insertion: for each resource written by an async compute pass and read by a subsequent graphics pass, insert `SyncPoint { resource, compute_pass_index, graphics_pass_index }`. Wire sync points as wgpu barriers between encoders | S1-G7 | H | T-FG-5.2 |
| T-FG-5.4 | Implement feature gating: check `wgpu::Features::TIMELINE_SEMAPHORE` at compile time. If absent, set `async_timeline = None` and log "Async compute not available on this device" | S1-G7 | L | T-FG-5.2 |
| T-FG-5.5 | Implement serial fallback: when `async_timeline` is `None`, flatten async-eligible compute passes onto the graphics timeline at their dependency-respecting positions. Barriers are already correct for this ordering | S1-G7 | L | T-FG-5.4 |
| T-FG-5.6 | Add `QueueType { Graphics, Compute }` and `SyncPoint` to `ScheduledPass` and `CompiledFrameGraph` respectively | S1-G7 | L | T-FG-5.2 |
| T-FG-5.7 | Unit tests: compute passes with no RAW on graphics (eligible), compute passes with RAW (ineligible), mixed eligibility, sync point placement, serial fallback produces same barrier set | S1-G7 | M | T-FG-5.1, T-FG-5.2, T-FG-5.3 |
| T-FG-5.8 | Acceptance: Async-eligible compute passes identified correctly. Sync points cover all cross-timeline dependencies. Serial fallback produces correct rendering. Async pass count reported in `CompilationResult` | S1-G7 | -- | All above |

---

## Phase 6: Dead Pass Elimination (Covers S1-G8)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-6.1 | Implement live output set definition: swap chain (always live), history resources (always live), and debug output resources (live when feature enabled). Produce `Vec<ResourceHandle>` | S1-G8 | L | T-FG-1.1 |
| T-FG-6.2 | Implement reverse reachability BFS: from live output set, find the producing pass for each live resource. Mark it live. Recursively mark all passes producing its inputs as live. All unmarked passes are dead | S1-G8 | M | T-FG-6.1, T-FG-2.1 |
| T-FG-6.3 | Implement pass culling: remove dead passes from the compiled plan. Remove resources exclusively used by dead passes from the allocation table. Re-index remaining passes | S1-G8 | M | T-FG-6.2 |
| T-FG-6.4 | Implement dynamic culling for debug/toggleable passes: at execution time, check a frame-level `FeatureSet` bitfield. A debug pass is live only if its feature bit is set. Dead passes are skipped during execution (not removed from the graph) | S1-G8 | M | T-FG-6.2 |
| T-FG-6.5 | [x] Add culling statistics to `CompilationResult`: `live_pass_count`, `culled_pass_count`, `estimated_gpu_time_saved_ms` | S1-G8 | L | T-FG-6.3 |
| T-FG-6.6 | Unit tests: pass with no consumer (dead), transitive liveness (A→B→C where C is live), diamond liveness (A→B, A→C, B→D, C→D where D is live), all-dead, all-live, dynamic toggling | S1-G8 | M | T-FG-6.2, T-FG-6.3 |
| T-FG-6.7 | Acceptance: Unused passes correctly removed. 3-5 passes culled in standard frame with debug disabled. Dynamic culling toggles in <1ms. Culling statistics reported | S1-G8 | -- | All above |

---

## Phase 7: Bridge + Emit (Covers S1-G3)

| Task ID | Description | Gap | Effort | Dependencies |
|---------|-------------|-----|--------|-------------|
| T-FG-7.1 | Implement Type channel: PyO3 `type_register()` function accepting `(component_id, component_name, field_layouts, flags)`. Populates Rust `TypeRegistry`. Called from Python `ComponentMeta.__new__()` | S1-G3 | M | T-FG-1.1 |
| T-FG-7.2 | Add Type channel `frame_graph_compile()` PyO3 function: accepts `Vec<PyPassNode>`, `Vec<PyResourceDesc>`, runs all 7 compiler phases, returns `PyCompilationResult` | S1-G3 | H | T-FG-1.6, T-FG-2.2, T-FG-3.7, T-FG-4.6, T-FG-5.6, T-FG-6.5 |
| T-FG-7.3 | Implement Data channel: PyO3 `component_read()` and `component_write()` functions. Read acquires read lock, indexes archetype column at offset, returns raw bytes converted to Python type. Write acquires write lock, writes bytes at offset | S1-G3 | M | T-FG-1.1 |
| T-FG-7.4 | Implement Data channel: PyO3 `world_spawn()`, `world_despawn()`, `world_query()` functions. Spawn creates entity in component store, returns entity ID. Despawn frees row. Query returns matching entity IDs | S1-G3 | M | T-FG-7.3 |
| T-FG-7.5 | Implement Command channel: crossbeam SPSC queue `bounded::<RendererCommand>(16)`. PyO3 functions `renderer_resize()`, `renderer_screenshot()`, `renderer_recompile_materials()`, `renderer_shutdown()` enqueue commands. Render thread drains at frame start | S1-G3 | M | None |
| T-FG-7.6 | Build `CompiledFrameGraph` emit: assemble passes, resource allocations, barrier batches, async timeline, and culling stats into the final struct. Implement `Debug` and `Serialize` for golden file output | S1-G3 | M | T-FG-7.2 |
| T-FG-7.7 | Implement Python-side `CompilationResult` class receiving the Rust result. Fields: `success: bool`, `pass_count`, `culled_count`, `async_pass_count`, `memory_savings_percent`, `errors: Vec<CompileError>` | S1-G3 | M | T-FG-7.6 |
| T-FG-7.8 | Implement atomic swap of `CompiledFrameGraph` on recompile: `ArcSwap<CompiledFrameGraph>`. Python calls `compile()` which produces a new graph; the runtime atomically swaps the reference. In-flight frames continue using the old graph | S1-G3 | H | T-FG-7.6 |
| T-FG-7.9 | Add debugging capabilities: `impl fmt::Display for CompiledFrameGraph` showing pass order, barrier count, alias groups. Add `--dump-frame-graph` CLI flag for compiler output inspection | S1-G3 | M | T-FG-7.6 |
| T-FG-7.10 | Acceptance: All 3 channels operational. Type channel delivers type layouts. Data channel reads/writes fields at <100ns per call. Command channel delivers commands with <1ms latency. CompiledFrameGraph serializes to JSON for golden file testing | S1-G3 | -- | All above |

---

## GAP COVERAGE SUMMARY

| Gap ID | Gap Name | Severity | Covered By |
|--------|----------|----------|------------|
| S1-G1 | Frame Graph Compiler | CRITICAL | T-FG-1.1, T-FG-1.2, T-FG-1.3, T-FG-1.6, T-FG-1.7, T-FG-1.8, T-FG-7.2 |
| S1-G2 | View Trait | CRITICAL | T-FG-1.4, T-FG-1.5 |
| S1-G3 | Bridge Channel Protocol | CRITICAL | T-FG-7.1 through T-FG-7.10 |
| S1-G4 | Dependency Analysis | HIGH | T-FG-2.1 through T-FG-2.7 |
| S1-G5 | Resource Aliasing | HIGH | T-FG-3.1 through T-FG-3.9 |
| S1-G6 | Automatic Barrier Insertion | HIGH | T-FG-4.1 through T-FG-4.8 |
| S1-G7 | Async Scheduling | MEDIUM | T-FG-5.1 through T-FG-5.8 |
| S1-G8 | Dead Pass Elimination | MEDIUM | T-FG-6.1 through T-FG-6.7 |

**Total tasks: 47**
- Phase 1: 8 tasks
- Phase 2: 7 tasks
- Phase 3: 9 tasks
- Phase 4: 8 tasks
- Phase 5: 8 tasks
- Phase 6: 7 tasks
- Phase 7: 10 tasks (including integration)
