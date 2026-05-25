# PHASE 3 TODO: Barrier Execution Backend

## T-FG-3.1: Implement execute_barriers() in MockContext

**File**: `tests/framegraph/mock_context.py`

**Tasks**:
- [ ] Implement `execute_barriers(barriers)` recording all barriers
- [ ] Add `get_barrier_log() -> list[Barrier]` for test assertions
- [ ] Track barrier order relative to begin_pass/end_pass calls
- [ ] Add `assert_barrier_before_pass(barrier_filter, pass_name)` helper

**Acceptance Criteria**:
- Tests can verify correct barriers emitted for each pass
- Barrier ordering is captured and assertable
- No actual GPU commands (mock only records)

---

## T-FG-3.2: Wire BarrierManager into Execute Loop

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] In `execute()`, call `barrier_manager.analyze_pass(pass_node)` before each pass
- [ ] Call `context.execute_barriers(batch)` with resulting batch
- [ ] Remove TODO comment at lines 664-670 — replace with real call
- [ ] Add error handling for barrier execution failures

**Acceptance Criteria**:
- `execute()` no longer logs/tracks barriers — it executes them
- Each pass has barriers analyzed and executed before `begin_pass()`
- Barrier failures raise `BarrierExecutionError` with pass context

---

## T-FG-3.3: Implement State-to-Native Mapping

**File**: `engine/rendering/rhi/barrier_mapper.py` (new)

**Tasks**:
- [ ] Create `map_state_to_wgpu(state: ResourceState) -> wgpu.TextureUsage`
- [ ] Create `map_stage_to_wgpu(stage: PipelineStage) -> wgpu.ShaderStages`
- [ ] Handle all ResourceState enum values
- [ ] Handle all PipelineStage enum values

**Acceptance Criteria**:
- Every abstract state maps to valid wgpu usage
- Unmapped states raise clear error (not silent fallback)
- Unit tests cover all enum values

---

## T-FG-3.4: Implement execute_barriers() in WgpuContext

**File**: `engine/rendering/rhi/wgpu_context.py`

**Tasks**:
- [ ] Import barrier_mapper for state conversion
- [ ] Implement `execute_barriers(barriers)` using wgpu command encoder
- [ ] Handle TRANSITION barriers via texture/buffer barriers
- [ ] Handle UAV barriers via memory barriers (same layout, different access)
- [ ] Handle ALIASING barriers via discard/ownership transfer

**Acceptance Criteria**:
- wgpu command encoder receives correct barrier commands
- GPU debugger (RenderDoc) shows barriers at expected points
- All three barrier types (TRANSITION, UAV, ALIASING) handled

---

## T-FG-3.5: Aliasing Barrier Integration

**File**: `engine/rendering/framegraph/barrier_manager.py`

**Tasks**:
- [ ] Track active alias group member per group
- [ ] Emit aliasing barrier when active member changes
- [ ] Call `context.execute_barriers()` for aliasing barriers separately or batched

**Acceptance Criteria**:
- First use of aliased resource after previous member emits aliasing barrier
- Aliasing barriers appear in MockContext log
- No aliasing barrier if same member used consecutively

---

## T-FG-3.6: Validation Layer for Barriers

**File**: `engine/rendering/rhi/validation_context.py` (new)

**Tasks**:
- [ ] Create `ValidationContext` wrapping real context
- [ ] In `execute_barriers()`, verify state transitions are valid
- [ ] Log warnings for redundant barriers (same state transition twice)
- [ ] Error on missing barriers (detected via state tracker comparison)

**Acceptance Criteria**:
- ValidationContext catches invalid transitions before GPU
- Redundant barriers logged but not failed
- Missing barrier detection works by comparing expected vs. actual states

---

## Definition of Done

- All tasks checked
- `uv run pytest tests/framegraph -v` passes with barrier verification
- wgpu backend executes barriers (visible in RenderDoc)
- ValidationContext catches intentionally broken barrier sequences
- TODO comment at frame_graph.py:664-670 removed
