# CLARIFICATION: engine/rendering/framegraph

## Philosophical Framing

The frame graph is the **declarative GPU command scheduler** — it transforms a high-level description of render passes and resource dependencies into an optimal execution plan. This abstraction exists because modern GPUs are massively parallel devices with complex memory hierarchies and multiple command queues, yet shader authors and render engineers want to think in terms of "render the scene, then apply post-processing" rather than "insert a barrier after the depth pass with stage=LATE_FRAGMENT_TESTS, access=DEPTH_STENCIL_ATTACHMENT_WRITE before the lighting pass begins."

## Why Frame Graphs Exist

Traditional "immediate mode" rendering APIs (OpenGL, early D3D) hid synchronization complexity but paid for it with implicit pipeline stalls and memory waste. Modern APIs (Vulkan, D3D12, Metal) expose the true hardware model: explicit barriers, manual resource lifetimes, multi-queue execution. Frame graphs restore productivity while preserving performance by:

1. **Capturing the render DAG** — passes declare what they read/write, not when/how
2. **Computing optimal barriers** — state machine tracks each resource's current state
3. **Finding aliasing opportunities** — transient resources with non-overlapping lifetimes share memory
4. **Scheduling async compute** — independent compute work overlaps with graphics

## Design Rationale

### Why Python for the High-Level Graph?

The frame graph is constructed once per frame architecture (not per frame), making Python's expressiveness worth more than raw speed. The actual execution path is:

```
Python (build graph) -> JSON IR -> Rust (compile) -> GPU commands
```

Python handles the declarative construction; Rust handles the low-latency execution. The `serialize()` method is the FFI boundary.

### Why Producer/Consumer Dependency Tracking?

The `_build_dependency_graph()` algorithm tracks which pass produces each resource and which passes consume it. This is the minimal information needed to:
- Establish execution ordering (topological sort)
- Identify dead passes (no consumer for their outputs)
- Insert barriers at the correct points

Alternative approaches (explicit dependency declaration, resource-centric graphs) were rejected because they require render engineers to manually track information the system can derive automatically.

### Why Lifetime-Based Aliasing?

Memory aliasing is safe when two resources never coexist. The `first_use_pass` and `last_use_pass` indices define each transient resource's lifetime. Non-overlapping lifetimes can share the same GPU memory allocation.

This is a greedy first-fit algorithm (assign to first compatible alias group). Optimal bin-packing is NP-hard; the greedy approach achieves ~90% of optimal with O(n*m) complexity where n=resources, m=groups.

### Why State Machine Barriers?

The barrier manager maintains a `ResourceStateTracker` that knows each resource's current state (SHADER_READ, RENDER_TARGET, UNORDERED_ACCESS, etc.). When a pass requires a different state, a transition barrier is generated.

This is the D3D12/Vulkan model. The state machine ensures:
- Transitions happen exactly once per state change
- Pipeline stages are matched to access types
- UAV hazards (read-after-write to same UAV) get explicit barriers

### Why Multi-Queue Scheduling?

Modern GPUs have independent graphics, compute, and copy queues. Independent work can execute in parallel across queues, but cross-queue dependencies require explicit fence synchronization.

The `async_scheduler.py` module:
1. Identifies passes that can safely run on compute queue
2. Groups consecutive async compute work
3. Inserts sync points where compute results flow to graphics

The `estimate_overlap_benefit()` heuristic prevents over-scheduling — async compute has overhead; only parallelize when the benefit exceeds the sync cost.

## The RHI Boundary

The current implementation is "real but unplugged." All algorithms execute correctly; the results are captured in data structures but not sent to the GPU. The critical comment (frame_graph.py:664-670) explains:

```python
# In a real implementation, this would call into the RHI
# to execute the actual GPU barriers.
# The context object should provide a method like:
#   context.execute_barriers(batch.barriers)
# For now, we log/track the barriers for debugging purposes.
```

The `context: Any` parameter is the integration point. Completing the frame graph means:
1. Defining a `RHIContext` protocol with `execute_barriers()`, `allocate_resource()`, etc.
2. Implementing that protocol for wgpu (primary backend)
3. Calling the protocol methods instead of logging

## What "Complete" Looks Like

A complete frame graph:
- Builds passes via method chaining API (existing)
- Compiles with dependency analysis, culling, aliasing (existing)
- Executes barriers via RHI (TODO)
- Allocates transient memory via RHI (TODO)
- Submits command buffers per queue (TODO)
- Waits on cross-queue fences (TODO)

The Python code remains the source of truth for graph construction. Rust handles the hot path (actual GPU submission). The JSON IR is the stable contract between them.
