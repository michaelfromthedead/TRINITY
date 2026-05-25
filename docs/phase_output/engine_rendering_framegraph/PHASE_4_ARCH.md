# PHASE 4 ARCHITECTURE: Async Compute Execution

## Overview

Enable multi-queue GPU execution with real fence synchronization. The `AsyncScheduler` already computes queue assignments and sync points; this phase submits commands to multiple queues with fence-based ordering.

## Architecture Decisions

### ADR-FG-013: Queue Abstraction

**Decision**: RHIContext exposes named queues; scheduler assigns passes by name.

```python
class RHIContext(Protocol):
    def get_queue(self, queue_type: QueueType) -> QueueHandle: ...
    def submit_to_queue(self, queue: QueueHandle, commands: CommandBuffer, fences: list[FenceOp]) -> None: ...
```

**Rationale**:
- Not all GPUs have separate compute queue (mobile, Intel iGPU)
- Backend can map COMPUTE -> GRAPHICS if needed
- Scheduler remains queue-agnostic

**Consequences**:
- `get_queue(COMPUTE)` may return same handle as `get_queue(GRAPHICS)`
- Sync points between same queue become no-ops
- Scheduler's `estimate_overlap_benefit()` returns 0 when queues are same

### ADR-FG-014: Fence Values as Monotonic Timeline

**Decision**: Each queue has a monotonically increasing fence value; sync points reference (queue, value) pairs.

**Rationale**:
- D3D12/Vulkan timeline semaphores use this model
- Signal increments value; wait blocks until value reached
- No fence object allocation/deallocation per sync point

**Consequences**:
- `FenceOp.fence_value` is queue-local counter
- Backend tracks current fence value per queue
- Wait-before-signal is detected at submit time (error)

### ADR-FG-015: Command Buffer Per Pass Group

**Decision**: Consecutive passes on same queue share a command buffer; queue change triggers submit.

**Rationale**:
- Minimizes submit overhead (each submit has CPU cost)
- Sync points force submits (can't wait mid-recording)
- Matches `get_parallel_groups()` output from scheduler

**Consequences**:
- `parallel_groups` from scheduler define command buffer boundaries
- Each group is one command buffer on its queue
- Sync points trigger immediate submit of pending group

### ADR-FG-016: Overlap Estimation Drives Scheduling

**Decision**: If `estimate_overlap_benefit()` < threshold, don't use async compute.

**Rationale**:
- Async compute has overhead (sync points, queue management)
- Small compute passes may cost more to parallelize than inline
- Threshold tunable via `AsyncSchedulerConfig`

**Consequences**:
- Config `benefit_threshold` controls cutoff
- Passes below threshold execute on graphics queue
- Estimation is conservative (prefers graphics when uncertain)

## Execution Flow

```
Frame execution with async compute:

1. Scheduler computes queue assignments
2. For each parallel group:
   a. If GRAPHICS: record to graphics command buffer
   b. If COMPUTE: record to compute command buffer
3. At sync point:
   a. Submit pending command buffers with signal fence
   b. New command buffer waits on dependency's fence
4. Final submit: all command buffers with end-of-frame fence
5. CPU waits on frame completion fence (if needed)
```

## Component Diagram

```
+---------------------+
|  AsyncScheduler     |
|  - queue assignment |
|  - sync points      |
+----------+----------+
           |
           | ScheduledPass[]
           v
+----------+----------+
|  FrameGraph         |
|  execute()          |
+----------+----------+
           |
           | commands
           v
+-----+----+----+-----+
|     |         |     |
v     v         v     v
Graphics      Compute   Copy
Queue         Queue     Queue
  |             |
  +--- sync ----+
       (fence)
```

## Files Affected

- `engine/rendering/framegraph/async_scheduler.py` — no changes (already complete)
- `engine/rendering/framegraph/frame_graph.py` — use scheduler output in execute()
- `engine/rendering/framegraph/context.py` — add queue/submit methods to protocol
- `engine/rendering/rhi/wgpu_context.py` — implement multi-queue submission
- `engine/rendering/rhi/fence_manager.py` — new file for timeline fence tracking

## Integration Points

- `AsyncScheduler.schedule()` returns `list[ScheduledPass]`
- `ScheduledPass.queue` determines which queue
- `ScheduledPass.sync_before/sync_after` determine fence operations
- wgpu `Queue.submit()` with timeline semaphore (when available) or fallback
