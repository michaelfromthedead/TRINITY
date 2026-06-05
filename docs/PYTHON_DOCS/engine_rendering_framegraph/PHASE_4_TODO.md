# PHASE 4 TODO: Async Compute Execution

## T-FG-4.1: Add Queue Methods to RHIContext Protocol

**File**: `engine/rendering/framegraph/context.py`

**Tasks**:
- [ ] Add `get_queue(queue_type: QueueType) -> QueueHandle` to protocol
- [ ] Add `submit_to_queue(queue: QueueHandle, commands: CommandBuffer, fences: list[FenceOp])` to protocol
- [ ] Define `QueueHandle` type (opaque handle)
- [ ] Define `CommandBuffer` type (backend-specific, opaque to Python)

**Acceptance Criteria**:
- Protocol includes queue management methods
- QueueHandle is hashable for tracking
- CommandBuffer can be passed to submit

---

## T-FG-4.2: Implement Queue Dispatch in FrameGraph Execute

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] In `execute()`, get scheduled passes from `_async_scheduler.schedule()`
- [ ] Group consecutive passes by queue into command buffers
- [ ] Call `submit_to_queue()` at sync point boundaries
- [ ] Insert FenceOp.signal after each submit
- [ ] Insert FenceOp.wait before dependent submit

**Acceptance Criteria**:
- Passes execute on their assigned queues
- Sync points translate to fence signal/wait
- Single-queue fallback works (all passes on graphics)

---

## T-FG-4.3: Implement FenceManager

**File**: `engine/rendering/rhi/fence_manager.py` (new)

**Tasks**:
- [ ] Create `FenceManager` class tracking fence values per queue
- [ ] Implement `next_fence_value(queue: QueueType) -> int`
- [ ] Implement `signal(queue: QueueType) -> FenceOp`
- [ ] Implement `wait_for(queue: QueueType, value: int) -> FenceOp`
- [ ] Validate wait-before-signal at generation time

**Acceptance Criteria**:
- Fence values are monotonically increasing per queue
- Attempting to wait on future (unsignaled) value raises error
- FenceOps are correctly typed for protocol

---

## T-FG-4.4: Implement Multi-Queue in WgpuContext

**File**: `engine/rendering/rhi/wgpu_context.py`

**Tasks**:
- [ ] Implement `get_queue()` returning appropriate wgpu queue
- [ ] Handle fallback: if device has no compute queue, return graphics
- [ ] Implement `submit_to_queue()` using wgpu Queue.submit()
- [ ] Integrate FenceManager for sync operations

**Acceptance Criteria**:
- wgpu submission works on multiple queues (if available)
- Fallback to single queue works on limited hardware
- Fences translate to wgpu timeline semaphores (or polyfill)

---

## T-FG-4.5: Parallel Group Command Buffer Batching

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Use `_async_scheduler.get_parallel_groups()` for batching
- [ ] Create one command buffer per parallel group
- [ ] Submit batch with appropriate sync operations
- [ ] Track which passes are in which command buffer

**Acceptance Criteria**:
- Consecutive async compute passes share command buffer
- Group boundaries trigger submit
- Fewer submits than passes (batching verified)

---

## T-FG-4.6: Overlap Benefit Threshold

**File**: `engine/rendering/framegraph/async_scheduler.py`

**Tasks**:
- [ ] Add `benefit_threshold: float` to AsyncSchedulerConfig
- [ ] In `_can_run_async()`, check `estimate_overlap_benefit() >= threshold`
- [ ] Default threshold to conservative value (e.g., 0.1 = 10% improvement)
- [ ] Add config override for testing/tuning

**Acceptance Criteria**:
- Small compute passes stay on graphics queue
- Threshold configurable per-frame-graph
- Metrics show overlap benefit vs. threshold

---

## T-FG-4.7: MockContext Multi-Queue Support

**File**: `tests/framegraph/mock_context.py`

**Tasks**:
- [ ] Implement `get_queue()` returning mock handles
- [ ] Implement `submit_to_queue()` recording submissions
- [ ] Track fence values per queue
- [ ] Add `assert_queue_submissions()` for test verification

**Acceptance Criteria**:
- Tests can verify multi-queue execution
- Fence ordering verified in tests
- Can simulate single-queue device (same handle for all types)

---

## Definition of Done

- All tasks checked
- `uv run pytest tests/framegraph -v` passes with async compute tests
- Multi-queue execution verified on wgpu backend
- Single-queue fallback tested (simulated limited hardware)
- Overlap benefit threshold prevents over-parallelization
