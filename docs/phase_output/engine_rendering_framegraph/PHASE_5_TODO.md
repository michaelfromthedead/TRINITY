# PHASE 5 TODO: Rust Bridge Validation

## T-FG-5.1: Add Schema Version to Serialize Output

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Add `"version": "1.0"` to serialize() output
- [ ] Document schema version in module docstring
- [ ] Add version check in future deserializers

**Acceptance Criteria**:
- JSON output includes version field
- Version is semantic (major.minor)
- Rust side can validate version before parsing

---

## T-FG-5.2: Validate serialize() Before Allocation

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] In `serialize()`, check all resources have allocations
- [ ] Raise `SerializationError` if any allocation is pending
- [ ] Include resource name in error message

**Acceptance Criteria**:
- `serialize()` before `begin_frame()` raises clear error
- Error identifies which resource lacks allocation
- After `begin_frame()`, serialize succeeds

---

## T-FG-5.3: Complete IrResourceAccess Serialization

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Ensure `ResourceAccess` serializes all fields
- [ ] Include: resource name, state, subresource, access_flags
- [ ] Match Rust `IrResourceAccess` field names exactly

**Acceptance Criteria**:
- Python ResourceAccess -> JSON -> Rust IrResourceAccess round-trips
- No data loss in serialization
- Subresource null when not specified

---

## T-FG-5.4: Serialize SyncPoints for Rust

**File**: `engine/rendering/framegraph/frame_graph.py`

**Tasks**:
- [ ] Add sync_points array to serialize() output
- [ ] Include: signal_queue, wait_queue, fence_value
- [ ] Match Rust `IrSyncPoint` struct

**Acceptance Criteria**:
- Cross-queue sync points appear in JSON
- Queue types serialized as strings matching Rust enum
- Fence values are integers

---

## T-FG-5.8: Document Schema Contract

**File**: `docs/architecture/framegraph_ir_schema.md` (if requested)

**Tasks**:
- [ ] Document JSON schema for IrFrameGraph
- [ ] Include example JSON
- [ ] Note version compatibility rules

**Acceptance Criteria**:
- Schema documented with all fields
- Example matches actual serialize() output
- Created only if explicitly requested

---

## Definition of Done

- All tasks checked
- `uv run pytest tests/framegraph/test_serialization.py -v` passes
- Rust parses Python output without error
- Schema version present in output
- No `deny_unknown_fields` violations
