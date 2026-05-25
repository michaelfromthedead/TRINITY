# PHASE 1 TODO: RHI Context Protocol

## T-FG-1.1: Define RHIContext Protocol

**File**: `engine/rendering/framegraph/context.py` (new)

**Tasks**:
- [ ] Create `RHIContext` as `typing.Protocol`
- [ ] Define `execute_barriers(barriers: Sequence[Barrier]) -> None`
- [ ] Define `allocate_transient(desc: ResourceDescriptor) -> AllocationHandle`
- [ ] Define `begin_pass(pass_node: PassNode) -> None`
- [ ] Define `end_pass(pass_node: PassNode) -> None`
- [ ] Define `submit_queue(queue: QueueType, fences: Sequence[FenceOp]) -> None`
- [ ] Define `AllocationHandle` as opaque type (NewType or class)
- [ ] Define `FenceOp` dataclass with operation, fence_value, target_queue

**Acceptance Criteria**:
- Protocol is importable: `from engine.rendering.framegraph import RHIContext`
- MyPy validates protocol method signatures
- No runtime dependencies outside typing and dataclasses

---

## T-FG-1.2: Update FrameGraph Type Annotations

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Import `RHIContext` from `context.py`
- [ ] Change `execute(context: Any)` to `execute(context: RHIContext)`
- [ ] Add type annotations to internal context usage (lines 664-670 area)
- [ ] Update docstrings to reference protocol methods

**Acceptance Criteria**:
- `context: Any` no longer appears in frame_graph.py
- MyPy passes with strict mode
- Existing tests still pass (context mock must satisfy protocol)

---

## T-FG-1.3: Update BarrierManager Type Annotations

**File**: `engine/rendering/framegraph/barrier_manager.py`

**Tasks**:
- [ ] Import `RHIContext` from `context.py`
- [ ] Update any context-accepting methods to use `RHIContext`
- [ ] Ensure `BarrierBatch` is compatible with `execute_barriers()` signature

**Acceptance Criteria**:
- Barrier execution path is fully typed
- `Barrier` dataclass fields match protocol expectations

---

## T-FG-1.4: Export Protocol from Package

**File**: `engine/rendering/framegraph/__init__.py`

**Tasks**:
- [ ] Add `RHIContext` to `__all__`
- [ ] Add `AllocationHandle` to `__all__`
- [ ] Add `FenceOp` to `__all__`

**Acceptance Criteria**:
- `from engine.rendering.framegraph import RHIContext, AllocationHandle, FenceOp` works

---

## T-FG-1.5: Create MockContext for Testing

**File**: `tests/framegraph/mock_context.py` (new)

**Tasks**:
- [ ] Implement `MockContext` satisfying `RHIContext` protocol
- [ ] Record all method calls for assertion
- [ ] Add helpers: `assert_barriers_executed()`, `assert_queue_submitted()`
- [ ] Default to no-op behavior (no exceptions)

**Acceptance Criteria**:
- MockContext passes isinstance check (Protocol structural match)
- Existing frame graph tests work with MockContext
- Can assert specific barrier/fence sequences were requested

---

## Definition of Done

- All tasks checked
- `uv run python -m mypy engine/rendering/framegraph --strict` passes
- `uv run pytest tests/framegraph -v` passes
- No `Any` types in context-related code paths
