# GAPSET_2_FRAME_GRAPH -- Clarifications

## Architectural Divergence: Two Independent Implementations

The most significant finding from source-code investigation is that the frame graph subsystem consists of **two parallel, independent implementations**:

### Rust Compiler (`renderer-backend`)
- **Location**: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` (3,156 lines)
- **Builder**: GAPSET_3_BRIDGE built this compiler
- **Phases**: 1 (IR), 2 (DAG + topological sort), 3 (lifetimes), 4 (barriers), 5 (async), 6 (dead pass elimination)
- **Strength**: Production-quality DAG builder and Kahn's topological sort with cycle detection
- **Design**: Owned IR types, standalone compilation, JSON statistics output

### Python Frame Graph (`engine/rendering/framegraph/`)
- **Location**: 7 modules (`__init__.py`, `frame_graph.py` (880 lines), `pass_node.py` (727 lines), `resource_manager.py` (655 lines), `barrier_manager.py` (575 lines), `async_scheduler.py` (480 lines), `config.py` (61 lines))
- **Phases**: All 7 (IR + all 6 compiler phases + bridge serialization)
- **Strength**: Strong resource management with handle-based system, comprehensive barrier state tracking with pipeline stage mapping, proper async scheduling with sync point computation
- **Design**: Dataclass-based, `PassNode` ABC with concrete subclasses, `ResourceManager`/`BarrierManager`/`AsyncScheduler` as service classes

### How They Connect

The only bridge between Python and Rust is **JSON serialization**:
1. Python `FrameGraph.serialize()` (frame_graph.py line 778) produces a JSON dict with `passes` and `resources` arrays
2. Rust `deserialize_from_json()` (mod.rs line 1761) parses that JSON into `Vec<IrPass>` and `Vec<IrResource>`
3. Rust `execute()` (mod.rs line 1998) compiles and returns JSON statistics

This is **not** the PyO3 FFI assumed by the TODO plan. There is:
- No `type_register()` PyO3 function
- No `frame_graph_compile()` PyO3 entry point
- No `component_read()`/`component_write()` Data channel
- No crossbeam SPSC Command channel

## Divergence from TODO Plan

### 1. No Unified Compiler Pipeline
The TODO plan (T-FG-1.6, T-FG-7.2) assumes a PyO3 bridge where Python `compile()` serializes passes and calls into Rust. Reality: Python `compile()` runs a full Python-only compilation. Rust `execute()` is a standalone function callable from Rust tests or a potential PyO3 wrapper.

### 2. View Trait vs ViewType Enum
The TODO (T-FG-1.4, T-FG-1.5) specifies a `View` trait with `fn bind()` returning bind groups. Reality: `ViewType` is a 9-variant enum in both implementations. No trait object, no `bind()` method, no `EmptyView`/`CameraView`. The enum approach is simpler but lacks the extensibility and encapsulation of the trait approach.

### 3. Bridge: JSON vs PyO3
The TODO (T-FG-7.1 through T-FG-7.10) specifies a 3-channel protocol (Type, Data, Command) using PyO3 FFI. Reality: a single JSON serialization channel exists. The 3-channel design may be over-engineered for the current use case, but the JSON bridge provides no ECS integration, no type reflection, and no command queuing.

### 4. Graph Coloring vs Interval Heuristic
The TODO (T-FG-3.3) specifies greedy largest-first graph coloring. Reality: Python uses an interval-overlap heuristic (sort by first_use_pass, assign to first compatible group). This is simpler and may achieve similar memory savings but lacks theoretical optimality guarantees.

### 5. Topological Sort Completeness
The TODO (T-FG-2.4, T-FG-2.5) specifies topological depth assignment and parallel region identification. Neither is implemented. The currently implementation produces a valid execution order but cannot identify which passes could execute in parallel.

### 6. Acceptance Tests
Several acceptance tasks are marked [~] or [-] because the tests they describe don't exist:
- T-FG-1.8 (10-pass compile): no such test
- T-FG-2.7 (10-pass/20-edge benchmark): not run
- T-FG-3.9 (40%+ memory savings): impossible to verify without wgpu allocator
- T-FG-4.8 (72 transition pairs): not tested
- T-FG-6.7 (3-5 passes culled): no such benchmark

## Cross-References to GAPSET_3_BRIDGE

### What GAP 3 Built
GAPSET_3_BRIDGE built the full Rust frame graph compiler at `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`. This compiler implements all 6 phases and is the **production target** that GAPSET_2 should ultimately use.

### What GAP 2 Should Reference
GAPSET_2 tasks that say "implement" should first check whether GAP 3 already provides the implementation:

| Task | GAP 3 Status | Recommendation |
|------|-------------|---------------|
| T-FG-1.1 | Fully implemented [x] | No work needed -- use existing `IrPass`/`IrResource`/`IrEdge` |
| T-FG-1.2 | Fully implemented [x] | No work needed -- `TryFrom<PyPassNode>` exists |
| T-FG-1.4 | Absent [-] | New work: implement `View` trait |
| T-FG-2.1 | Fully implemented [x] | No work needed -- `build_dag()` exists |
| T-FG-2.2 | Fully implemented [x] | No work needed -- `topological_sort()` exists |
| T-FG-3.1 | Fully implemented [x] | No work needed -- `compute_lifetimes()` exists |
| T-FG-4.1 | Fully implemented [x] | No work needed -- barrier computation exists |
| T-FG-5.1 | Fully implemented [x] | No work needed -- `async_schedule()` exists |
| T-FG-6.1 | Partial [~] | Extend: add explicit live output set definition |
| T-FG-7.2 | Partial [~] | Bridge exists as JSON, not PyO3 -- decide which direction |

### Key Design Decisions for Future Work

1. **Extend Rust or Python?** -- The Rust compiler is more complete for phases 2-6. The Python implementation is more complete for resource management (Phase 3), barrier management (Phase 4), and async scheduling (Phase 5). A unified approach should either (a) extend the Rust compiler with Python's stronger phases, or (b) wire the Python implementation through PyO3 to use the Rust DAG/barrier computation.

2. **JSON or PyO3?** -- The existing JSON bridge is functional but slow (serialization/deserialization overhead). PyO3 would give zero-copy access. Recommend JSON for prototype, PyO3 for production.

3. **View trait or ViewType enum?** -- The enum approach is simpler and sufficient for current needs. The trait approach would be needed for user-defined view types. Recommend sticking with enum until custom shader binding layouts are required.

4. **Single timeline or async?** -- Current async scheduling is a best-effort optimization. True async compute requires wgpu timeline semaphore support, which is platform-dependent. Recommend serial fallback as the default with async as an opt-in optimization.
