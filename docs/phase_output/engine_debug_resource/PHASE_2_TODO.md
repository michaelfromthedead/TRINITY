# PHASE 2 TODO - Integration (PENDING)

**Workflow:** RDC_WORKFLOW v1.2.0  
**Cluster:** engine_debug_resource  
**Status:** PENDING  
**Generated:** 2026-05-23

---

## Overview

Phase 2 tasks address integration gaps identified in the source investigations. All Phase 1 foundations are complete; these tasks wire them to external systems.

---

## 1. GPU Timing Integration

### Task GPU-INT-001: Add wgpu timestamp query support

**Priority:** HIGH  
**Estimate:** 4-6 hours  
**Dependencies:** wgpu Python bindings

**Description:**
Replace CPU-side `time.perf_counter_ns()` in GPUProfiler with wgpu GPU timestamp queries.

**Acceptance Criteria:**
- [ ] Create timestamp query set with sufficient capacity (256 queries)
- [ ] Create resolve buffer for timestamp readback
- [ ] Modify `begin_pass()` to write start timestamp via encoder
- [ ] Modify `end_pass()` to write end timestamp via encoder
- [ ] Add `resolve_frame()` to resolve queries and read buffer
- [ ] Convert GPU ticks to nanoseconds using device timestamp period
- [ ] Add fallback to CPU timing if timestamps unavailable
- [ ] Update unit tests to mock wgpu device

**Files to Modify:**
- `engine/debug/profiling/gpu.py`

---

## 2. Async I/O for Streaming

### Task STREAM-ASYNC-001: Add ThreadPoolExecutor-based streaming

**Priority:** HIGH  
**Estimate:** 4-6 hours  
**Dependencies:** None

**Description:**
Replace synchronous progress simulation with actual file I/O via ThreadPoolExecutor.

**Acceptance Criteria:**
- [ ] StreamManager accepts executor in constructor
- [ ] request_stream() submits file read to executor
- [ ] Callbacks invoked on completion with loaded bytes
- [ ] StreamState transitions correctly: PENDING -> ACTIVE -> COMPLETE
- [ ] Failed loads set StreamState.FAILED with error info
- [ ] Cancellation cancels pending futures
- [ ] Unit tests verify async behavior with mock executor

**Files to Modify:**
- `engine/resource/streaming/stream_manager.py`

---

### Task STREAM-ASYNC-002: Add progress callbacks

**Priority:** MEDIUM  
**Estimate:** 2-3 hours  
**Dependencies:** STREAM-ASYNC-001

**Description:**
Add progress reporting during async loads (useful for large assets).

**Acceptance Criteria:**
- [ ] request_stream() accepts optional progress_callback
- [ ] Progress callback invoked with (bytes_loaded, bytes_total)
- [ ] Works with chunked reads for large files
- [ ] UI can display loading progress

**Files to Modify:**
- `engine/resource/streaming/stream_manager.py`

---

### Task STREAM-ASYNC-003: Add asyncio alternative (optional)

**Priority:** LOW  
**Estimate:** 4-6 hours  
**Dependencies:** STREAM-ASYNC-001

**Description:**
Provide asyncio-based streaming as alternative to ThreadPoolExecutor.

**Acceptance Criteria:**
- [ ] AsyncStreamManager class using asyncio
- [ ] Uses aiofiles for async file operations
- [ ] Compatible with asyncio event loops
- [ ] Can coexist with ThreadPoolExecutor version

**Files to Create:**
- `engine/resource/streaming/async_stream_manager.py`

---

## 3. Budget Enforcement in Streaming

### Task BUD-STREAM-001: Wire BudgetManager to StreamManager

**Priority:** HIGH  
**Estimate:** 3-4 hours  
**Dependencies:** None

**Description:**
StreamManager checks budget before accepting stream requests.

**Acceptance Criteria:**
- [ ] BudgetAwareStreamManager wrapper class
- [ ] Checks BudgetManager.allocate() before streaming
- [ ] Returns None if over budget
- [ ] Frees budget if stream fails or is cancelled

**Files to Create:**
- `engine/resource/streaming/budget_aware_stream_manager.py`

**Files to Modify:**
- `engine/resource/streaming/__init__.py`

---

### Task BUD-STREAM-002: Trigger eviction when over budget

**Priority:** HIGH  
**Estimate:** 2-3 hours  
**Dependencies:** BUD-STREAM-001

**Description:**
When budget is exceeded, trigger eviction before rejecting stream request.

**Acceptance Criteria:**
- [ ] Check memory pressure via get_pressure()
- [ ] If pressure > threshold, call ResidencyManager.update() to evict
- [ ] Retry allocation after eviction
- [ ] Only reject if still over budget after eviction

**Files to Modify:**
- `engine/resource/streaming/budget_aware_stream_manager.py`

---

### Task BUD-STREAM-003: Wire ResidencyManager to streaming

**Priority:** HIGH  
**Estimate:** 2-3 hours  
**Dependencies:** BUD-STREAM-001

**Description:**
Update ResidencyManager state when streams complete.

**Acceptance Criteria:**
- [ ] request_residency() called when stream starts
- [ ] State transitions to RESIDENT when stream completes
- [ ] release_residency() called when asset unloaded
- [ ] touch() called when asset accessed

**Files to Modify:**
- `engine/resource/streaming/budget_aware_stream_manager.py`

---

## 4. Testing Framework Integration

### Task TEST-INT-001: Implement screenshot capture

**Priority:** MEDIUM  
**Estimate:** 2-3 hours  
**Dependencies:** Renderer framebuffer access

**Description:**
Screenshot action captures actual framebuffer instead of logging.

**Acceptance Criteria:**
- [ ] AutomationBot accepts renderer reference
- [ ] _do_screenshot() reads framebuffer pixels
- [ ] Saves to PNG using PIL/Pillow
- [ ] Screenshot path configurable
- [ ] Works headless (offscreen rendering)

**Files to Modify:**
- `engine/debug/testing/automation.py`

---

### Task TEST-INT-002: Implement checkpoint/restore

**Priority:** MEDIUM  
**Estimate:** 4-6 hours  
**Dependencies:** Game state serialization

**Description:**
Checkpoint action saves game state; restore action loads it.

**Acceptance Criteria:**
- [ ] AutomationBot accepts game state serializer
- [ ] _do_checkpoint() serializes current state
- [ ] _do_restore() deserializes saved state
- [ ] Supports multiple named checkpoints
- [ ] Warns if checkpoint doesn't exist on restore

**Files to Modify:**
- `engine/debug/testing/automation.py`

---

### Task TEST-INT-003: Wire InputSimulator to input system

**Priority:** MEDIUM  
**Estimate:** 3-4 hours  
**Dependencies:** Game input system

**Description:**
InputSimulator injects events into game input system instead of no-op.

**Acceptance Criteria:**
- [ ] InputSimulator accepts input system reference
- [ ] simulate_key() injects key events
- [ ] simulate_mouse_click() injects mouse events
- [ ] simulate_mouse_move() injects mouse motion
- [ ] simulate_gamepad_*() injects gamepad events
- [ ] Events processed by game on next frame

**Files to Modify:**
- `engine/debug/testing/automation.py`

---

## 5. Optional Enhancements

### Task OPT-001: Tracy profiler integration

**Priority:** LOW  
**Estimate:** 6-8 hours  
**Dependencies:** Tracy client library

**Description:**
Add Tracy profiler support for native C++/Rust profiling integration.

**Acceptance Criteria:**
- [ ] TracyProfiler class wrapping Tracy C API
- [ ] Zone begin/end via ctypes
- [ ] Frame markers
- [ ] Memory allocation tracking
- [ ] Can coexist with Python profiler

**Files to Create:**
- `engine/debug/profiling/tracy.py`

---

### Task OPT-002: Chrome Tracing export

**Priority:** LOW  
**Estimate:** 2-3 hours  
**Dependencies:** None

**Description:**
Export CPU profiler data to Chrome Tracing JSON format.

**Acceptance Criteria:**
- [ ] export_chrome_tracing() function
- [ ] Generates valid JSON for chrome://tracing
- [ ] Includes thread IDs
- [ ] Preserves hierarchy via stack depth

**Files to Create:**
- `engine/debug/profiling/export.py`

---

### Task OPT-003: Composite eviction policies

**Priority:** LOW  
**Estimate:** 3-4 hours  
**Dependencies:** None

**Description:**
Allow combining multiple eviction policies with weights.

**Acceptance Criteria:**
- [ ] CompositeEviction class
- [ ] Accepts list of (policy, weight) tuples
- [ ] Scores candidates by weighted sum
- [ ] Selects highest-scoring until bytes_needed

**Files to Modify:**
- `engine/resource/memory/eviction.py`

---

### Task OPT-004: Real tracemalloc integration

**Priority:** LOW  
**Estimate:** 2-3 hours  
**Dependencies:** None

**Description:**
Hook MemoryProfiler into Python's tracemalloc for automatic tracking.

**Acceptance Criteria:**
- [ ] Optional tracemalloc integration
- [ ] Automatic allocation tracking without manual calls
- [ ] Can still use manual tracking alongside

**Files to Modify:**
- `engine/debug/profiling/memory.py`

---

## Summary

| Category | Task ID | Priority | Status |
|----------|---------|----------|--------|
| GPU Timing | GPU-INT-001 | HIGH | PENDING |
| GPU Timing | GPU-INT-002 | HIGH | PENDING |
| Async I/O | STREAM-ASYNC-001 | HIGH | PENDING |
| Async I/O | STREAM-ASYNC-002 | MEDIUM | PENDING |
| Async I/O | STREAM-ASYNC-003 | LOW | PENDING |
| Budget | BUD-STREAM-001 | HIGH | PENDING |
| Budget | BUD-STREAM-002 | HIGH | PENDING |
| Budget | BUD-STREAM-003 | HIGH | PENDING |
| Testing | TEST-INT-001 | MEDIUM | PENDING |
| Testing | TEST-INT-002 | MEDIUM | PENDING |
| Testing | TEST-INT-003 | MEDIUM | PENDING |
| Optional | OPT-001 | LOW | PENDING |
| Optional | OPT-002 | LOW | PENDING |
| Optional | OPT-003 | LOW | PENDING |
| Optional | OPT-004 | LOW | PENDING |

**High Priority Tasks:** 7  
**Medium Priority Tasks:** 4  
**Low Priority Tasks:** 4  
**Total Tasks:** 15

---

## Execution Order

Recommended execution order based on dependencies:

1. **GPU-INT-001** (no dependencies, enables GPU profiling)
2. **STREAM-ASYNC-001** (no dependencies, enables async loading)
3. **BUD-STREAM-001** (no dependencies, enables budget checking)
4. **BUD-STREAM-002** (depends on BUD-STREAM-001)
5. **BUD-STREAM-003** (depends on BUD-STREAM-001)
6. **GPU-INT-002** (depends on GPU-INT-001)
7. **STREAM-ASYNC-002** (depends on STREAM-ASYNC-001)
8. **TEST-INT-001** through **TEST-INT-003** (independent, medium priority)
9. **Optional tasks** as time permits
