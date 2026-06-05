# PHASE 1 ARCHITECTURE: RHI Context Protocol

## Overview

Define the `RHIContext` protocol that replaces `context: Any` throughout the frame graph. This protocol is the contract between Python graph construction and GPU execution.

## Architecture Decisions

### ADR-FG-001: RHIContext as Protocol, Not ABC

**Decision**: Define `RHIContext` as a `typing.Protocol` rather than an abstract base class.

**Rationale**: 
- Protocols enable structural subtyping — any object with the right methods satisfies the protocol
- No inheritance required — wgpu backend can implement the protocol without coupling to frame graph module
- Duck typing with static type checking — MyPy validates without runtime isinstance checks

**Consequences**:
- Backend implementations don't import from frame graph
- Multiple backends can coexist (wgpu, mock, validation layer)
- Protocol evolution requires careful versioning

### ADR-FG-002: Method Granularity

**Decision**: Protocol methods match barrier/resource operations, not GPU commands.

```python
class RHIContext(Protocol):
    def execute_barriers(self, barriers: Sequence[Barrier]) -> None: ...
    def allocate_transient(self, desc: ResourceDescriptor) -> AllocationHandle: ...
    def begin_pass(self, pass_node: PassNode) -> None: ...
    def end_pass(self, pass_node: PassNode) -> None: ...
    def submit_queue(self, queue: QueueType, fences: Sequence[FenceOp]) -> None: ...
```

**Rationale**:
- Frame graph already batches barriers — protocol should accept batches
- Per-pass begin/end enables backend-specific render pass encoding
- Queue submission is the synchronization boundary

**Consequences**:
- Backend has freedom in command buffer management
- Frame graph doesn't dictate encoding strategy
- Debugging requires backend cooperation (begin/end_pass hooks)

### ADR-FG-003: Allocation Handle Opacity

**Decision**: `AllocationHandle` returned by `allocate_transient()` is opaque to Python.

**Rationale**:
- Python doesn't need to know GPU virtual addresses
- Handle is passed back to Rust via serialization or held for lifetime
- Enables deferred allocation (backend allocates on first use)

**Consequences**:
- Python can't directly access GPU memory (intentional)
- Debugging allocation requires backend introspection
- Handle must serialize correctly through JSON IR

### ADR-FG-004: Fence Operations as Data

**Decision**: Cross-queue synchronization expressed as `FenceOp` dataclass, not method calls.

```python
@dataclass
class FenceOp:
    operation: Literal["signal", "wait"]
    fence_value: int
    target_queue: QueueType
```

**Rationale**:
- Fence ops are data that can be serialized, logged, validated
- Backend batches fences with queue submission
- Async scheduler already computes sync points as data

**Consequences**:
- Protocol method is `submit_queue(..., fences)` not separate signal/wait
- Validation layer can verify fence ordering before GPU submission
- Matches D3D12/Vulkan fence model

## Component Diagram

```
+---------------------+
|   FrameGraph.py     |
|   (orchestrator)    |
+----------+----------+
           |
           | calls
           v
+----------+----------+
|   RHIContext        |  <-- typing.Protocol
|   (execution intf)  |
+----------+----------+
           ^
           | implements
           |
+----------+----------+     +-------------------+
|   WgpuContext       |     |   MockContext     |
|   (production)      |     |   (testing)       |
+---------------------+     +-------------------+
```

## Files Affected

- `engine/rendering/framegraph/context.py` — new file defining `RHIContext` protocol
- `engine/rendering/framegraph/frame_graph.py` — type annotation update (`context: RHIContext`)
- `engine/rendering/framegraph/barrier_manager.py` — type annotation update
- `engine/rendering/framegraph/__init__.py` — export `RHIContext`
