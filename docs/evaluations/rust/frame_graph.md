# Frame Graph Module Evaluation

**Module:** renderer-backend::frame_graph
**Location:** `/crates/renderer-backend/src/frame_graph/`
**Lines:** 26,915
**Quality Grade:** A

---

## Purpose

Intermediate representation (IR) and compiler for GPU render graphs. Transforms Python pass declarations into an optimized execution schedule with automatic barrier insertion, resource aliasing, and async compute scheduling.

---

## File Inventory

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| mod.rs | 10,626 | Core IR types, compiler phases | A |
| python.rs | 1,129 | PyPassNode → IrPass conversion | A |
| wgpu_barriers.rs | 982 | wgpu barrier generation | A- |
| async_tests.rs | 1,282 | Async scheduling tests | A |
| temp_edit.rs | 10,777 | Temp file (investigation artifact?) | ? |
| swap.rs | 549 | Double-buffered resource swap | A |
| debug_dumper.rs | 541 | GraphViz DOT export | A |
| type_bridge.rs | 537 | Type channel definitions | B |

---

## Core Types (mod.rs)

### Handle Types

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceHandle(pub u32);  // Identifies resources
pub struct PassIndex(pub usize);     // Identifies passes
```

### Pass Types

```rust
pub enum PassType {
    Graphics,    // Vertex + Fragment
    Compute,     // Compute dispatch
    Copy,        // Buffer/texture copy
    RayTracing,  // RT dispatch
}
```

### IrPass Structure

```rust
pub struct IrPass {
    pub name: String,
    pub pass_type: PassType,
    pub reads: ResourceAccessSet,
    pub writes: ResourceAccessSet,
    pub color_attachments: Vec<ColorAttachment>,
    pub depth_stencil: Option<DepthStencilAttachment>,
    pub instance_source: InstanceSource,
    pub dispatch_source: Option<DispatchSource>,
    pub view_type: ViewType,
}
```

### IrResource Structure

```rust
pub struct IrResource {
    pub handle: ResourceHandle,
    pub name: String,
    pub desc: ResourceDesc,
    pub is_external: bool,
    pub is_history: bool,
}
```

### IrEdge Structure

```rust
pub struct IrEdge {
    pub from: PassIndex,
    pub to: PassIndex,
    pub resource: ResourceHandle,
    pub edge_type: EdgeType,  // RAW, WAR, WAW
}
```

---

## Compiler Phases

### Phase 1: IR Construction
- PyPassNode → IrPass via TryFrom trait
- Validation: empty names, zero attachments, invalid handles
- Resource desc extraction

### Phase 2: DAG Building
- `build_dag()` - Produces IrEdge records
- Classifies edges as RAW/WAR/WAW
- HashSet deduplication

### Phase 3: Topological Sort
- `topological_sort()` - Kahn's algorithm
- BFS with VecDeque
- Cycle detection with descriptive error

### Phase 4: Resource Lifetimes
- `compute_lifetimes()` - (first_pass, last_pass) per resource
- Considers load_op, store_op for attachments
- Used for aliasing

### Phase 5: Barrier Generation
- `compute_barriers()` - State transition tuples
- Tracks resource state per pass
- Generates (from, to, before_state, after_state)

### Phase 6: Async Scheduling
- `async_schedule()` - Identifies async-eligible passes
- Compute/copy without RAW from graphics
- Returns eligible pass list

### Phase 7: Dead Pass Elimination
- `eliminate_dead_passes()` - Reverse reachability
- Preserves graphics (conservative)
- Returns pruned order

---

## Output Structure

```rust
pub struct CompiledFrameGraph {
    pub passes: Vec<IrPass>,
    pub resources: Vec<IrResource>,
    pub edges: Vec<IrEdge>,
    pub order: Vec<PassIndex>,
    pub barriers: Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>,
    pub async_passes: Vec<(PassIndex, String)>,
    pub eliminated_passes: Vec<PassIndex>,
}
```

---

## Test Coverage

**84 test files** in `/crates/renderer-backend/tests/`

| Test Category | Files | Coverage |
|---------------|-------|----------|
| Frame graph IR | 15 | Types, serialization, round-trip |
| Barrier generation | 5 | State transitions, batching |
| Async scheduling | 3 | Eligibility, sync points |
| Component store | 2 | SoA storage, queries |
| GPU tables | 3 | Mesh, material, texture |
| Noise shaders | 5 | WGSL validation |
| SDF shaders | 1 | Domain operations |
| Renderer | 1 | wgpu triangle |
| Integration | 49 | Various scenarios |

**Status:** Tests are comprehensive but **cannot compile** due to missing lib.rs exports.

---

## Blocking Issues

### 1. Not exported from lib.rs

```rust
// crates/renderer-backend/src/lib.rs is empty
// Need: pub mod frame_graph;
```

### 2. Python bridge is JSON, not PyO3

The compiler can serialize to JSON via `serde_json` but there's no PyO3 function to call it from Python directly.

### 3. wgpu barrier commands not generated

`wgpu_barriers.rs` has the mapping but actual `CommandEncoder` calls aren't wired.

---

## Recommendations

1. **Export from lib.rs** - Immediate
2. **Run tests** - Verify all pass after export
3. **Add PyO3 entry point** - `frame_graph_compile(json) -> json`
4. **Wire wgpu barriers** - Connect to actual command recording

---

## Python Counterpart

| Rust | Python | Status |
|------|--------|--------|
| IrPass | engine/rendering/framegraph/pass_node.py | Parallel |
| IrResource | engine/rendering/framegraph/resource_manager.py | Parallel |
| build_dag | frame_graph.py::_build_dependency_graph | Parallel |
| topological_sort | frame_graph.py::_topological_sort | Parallel |
| compute_barriers | barrier_manager.py | Parallel |
| async_schedule | async_scheduler.py | Parallel |

Both implementations exist independently. Python is active; Rust awaits PyO3 bridge.

---
