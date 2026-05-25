# PHASE 3 ARCHITECTURE: Barrier Execution Backend

## Overview

Wire the `BarrierManager` output to real GPU barrier commands. The barrier system already computes state transitions, pipeline stages, and access flags; this phase sends them to the GPU.

## Architecture Decisions

### ADR-FG-009: Barrier Batching at Queue Submission

**Decision**: Barriers execute as a batch at the start of each pass, not interleaved with draw calls.

**Rationale**:
- `BarrierBatch` already groups barriers per pass
- Modern GPUs batch barriers efficiently
- Interleaved barriers require render pass splitting (expensive)

**Consequences**:
- All barriers for a pass execute before `begin_pass()`
- Barrier-heavy frames pay upfront cost, not per-draw
- Validation layer can verify complete barrier coverage

### ADR-FG-010: Backend-Agnostic Barrier Representation

**Decision**: `Barrier` dataclass uses abstract states; backend maps to API-specific commands.

| Abstract State | wgpu | Vulkan | D3D12 |
|----------------|------|--------|-------|
| RENDER_TARGET | TextureUsage::RENDER_ATTACHMENT | COLOR_ATTACHMENT_OPTIMAL | RENDER_TARGET |
| SHADER_READ | TextureUsage::TEXTURE_BINDING | SHADER_READ_ONLY_OPTIMAL | PIXEL_SHADER_RESOURCE |
| UNORDERED_ACCESS | TextureUsage::STORAGE_BINDING | GENERAL | UNORDERED_ACCESS |

**Rationale**:
- Frame graph shouldn't know about Vulkan vs. D3D12
- Backend handles API mapping
- Same barrier logic works across all backends

**Consequences**:
- `Barrier` fields are portable (ResourceState, PipelineStage, AccessFlags)
- Backend implements `_map_state_to_native()`
- Debugging shows abstract states; backend logs show native

### ADR-FG-011: UAV Barrier as Explicit Command

**Decision**: UAV barriers (read-after-write to same UAV) emit explicit barrier commands, not state transitions.

**Rationale**:
- UAV hazard is not a state change — both sides are UNORDERED_ACCESS
- D3D12 requires explicit UAVBarrier
- Vulkan uses memory barrier with same layout

**Consequences**:
- `BarrierType.UAV` treated differently from `TRANSITION`
- Backend emits UAVBarrier or equivalent memory barrier
- `_check_uav_hazards()` output goes to dedicated code path

### ADR-FG-012: Aliasing Barriers for Memory Reuse

**Decision**: When alias group changes active member, emit aliasing barrier.

**Rationale**:
- Aliasing barriers (D3D12) / ownership transfer indicate memory reuse
- GPU must discard prior contents before new use
- Prevents undefined behavior from stale cache lines

**Consequences**:
- `create_aliasing_barrier()` already implemented
- Backend emits AliasingBarrier before new member's first use
- Barrier manager tracks which alias group member is "active"

## Execution Flow

```
Pass N start:
  1. BarrierManager.analyze_pass(pass_N)  -- computes needed barriers
  2. context.execute_barriers(batch)       -- GPU barriers execute
  3. context.begin_pass(pass_N)            -- render pass begins
  4. pass_N.execute(context)               -- user callback
  5. context.end_pass(pass_N)              -- render pass ends

If pass_N writes resource R:
  - State tracker updates R's state to pass_N's write state
  - Next pass reading R will see transition barrier if states differ
```

## Component Diagram

```
+---------------------+
|  BarrierManager     |
|  analyze_pass()     |
+----------+----------+
           |
           | BarrierBatch
           v
+----------+----------+
|  RHIContext         |
|  execute_barriers() |
+----------+----------+
           |
           | maps to
           v
+----------+----------+     +-------------------+
|  wgpu commands      |     |  Validation       |
|  - texture_barrier  |     |  - verify states  |
|  - buffer_barrier   |     |  - log transitions|
+---------------------+     +-------------------+
```

## Files Affected

- `engine/rendering/framegraph/barrier_manager.py` — call `context.execute_barriers()`
- `engine/rendering/framegraph/frame_graph.py` — wire barrier execution into pass loop
- `engine/rendering/rhi/wgpu_context.py` — implement `execute_barriers()` for wgpu
- `engine/rendering/rhi/barrier_mapper.py` — new file for state-to-native mapping

## Integration Points

- `BarrierBatch.barriers` is the input to `execute_barriers()`
- wgpu `CommandEncoder.insert_debug_marker()` for barrier visibility in GPU debuggers
- Validation layer can intercept barriers before backend
