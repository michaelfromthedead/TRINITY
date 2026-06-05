# PHASE 2 TODO: Resource Allocation Integration

## T-FG-2.1: Define Allocation Handle Structure

**File**: `engine/rendering/framegraph/context.py`

**Tasks**:
- [ ] Expand `AllocationHandle` to include `heap_id: int`, `offset: int`, `size: int`
- [ ] Add `is_pending: bool` field for deferred allocation state
- [ ] Add `memory_type: MemoryType` enum (DEVICE_LOCAL, HOST_VISIBLE, etc.)
- [ ] Ensure AllocationHandle is hashable for use in dicts

**Acceptance Criteria**:
- AllocationHandle can be used as dict key
- Fields accessible for debugging: `handle.heap_id`, `handle.offset`
- Serializable to JSON for Rust bridge

---

## T-FG-2.2: Update ResourceManager for Allocation Calls

**File**: `engine/rendering/framegraph/resource_manager.py`

**Tasks**:
- [ ] Add `_context: Optional[RHIContext]` field
- [ ] Add `bind_context(context: RHIContext)` method
- [ ] Modify `begin_frame()` to allocate alias group heaps via context
- [ ] Store `AllocationHandle` in `TransientResource.allocation`
- [ ] Implement `_allocate_alias_group(group_id)` helper

**Acceptance Criteria**:
- After `begin_frame()`, all transients have non-pending AllocationHandles
- Alias group members share same heap_id with different offsets
- History resources have two allocations (current/previous)

---

## T-FG-2.3: Implement Memory Type Inference

**File**: `engine/rendering/framegraph/resource_manager.py`

**Tasks**:
- [ ] Add `_infer_memory_type(desc: ResourceDescriptor) -> MemoryType`
- [ ] Map RENDER_TARGET, DEPTH_STENCIL -> DEVICE_LOCAL
- [ ] Map UPLOAD -> HOST_VISIBLE_WRITE_COMBINED
- [ ] Map READBACK -> HOST_VISIBLE_CACHED
- [ ] Default transients -> DEVICE_LOCAL

**Acceptance Criteria**:
- Memory type inference is deterministic
- All existing resource descriptors get valid memory types
- Unit test covers all usage flag combinations

---

## T-FG-2.4: Handle Allocation Failures

**File**: `engine/rendering/framegraph/resource_manager.py`

**Tasks**:
- [ ] Define `AllocationError` exception
- [ ] Catch context allocation failures in `begin_frame()`
- [ ] Provide diagnostic: which resource, how much memory, total budget
- [ ] Add `get_memory_budget() -> MemoryBudget` to RHIContext protocol

**Acceptance Criteria**:
- Out-of-memory raises `AllocationError` with actionable message
- Error includes resource name and requested size
- MockContext can simulate OOM for testing

---

## T-FG-2.5: History Resource Double-Buffering

**File**: `engine/rendering/framegraph/resource_manager.py`

**Tasks**:
- [ ] Ensure `HistoryResource` stores two AllocationHandles
- [ ] Implement `get_current_allocation() -> AllocationHandle`
- [ ] Implement `get_previous_allocation() -> AllocationHandle`
- [ ] Add `_swap_history_indices()` called by `begin_frame()`

**Acceptance Criteria**:
- History resource reports different handles for current vs. previous
- After `begin_frame()`, indices swap correctly
- Barrier manager can track both allocations' states

---

## T-FG-2.6: Serialize Allocations for Rust Bridge

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Update `serialize()` to include allocation handles in IR
- [ ] Add `"allocation": {"heap_id": X, "offset": Y, "size": Z}` to IrResource
- [ ] Handle pending allocations (error if serialized before begin_frame)

**Acceptance Criteria**:
- JSON output includes allocation data for all resources
- Rust `IrResource` type accepts allocation fields
- Serializing before allocation raises clear error

---

## Definition of Done

- All tasks checked
- `uv run pytest tests/framegraph -v` passes including allocation tests
- Memory type inference tested for all usage combinations
- History resource double-buffering verified
- Serialization includes allocation data
