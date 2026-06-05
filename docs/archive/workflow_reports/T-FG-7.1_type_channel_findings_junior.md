# T-FG-7.1 Type Channel (type_register + TypeRegistry) -- Junior QA Findings

**Reviewer**: Junior QA
**Review date**: 2026-05-23
**Files reviewed**:
- `crates/renderer-backend/src/frame_graph/type_bridge.rs` (implementation + whitebox tests)
- `crates/renderer-backend/src/type_registry.rs` (TypeRegistry backing store + tests)

**Test count**: 27 tests pass (16 type_bridge-specific + 11 TypeRegistry-specific).

**Build note**: The crate depends on `pyo3 0.22` which does not support Python 3.14 (the version on this host). The unit tests run via `--lib` (no PyO3 FFI needed) and pass cleanly. Integration/blackbox tests that require the full PyO3 module were not run.

---

## Summary

`type_register()` is a PyO3-exported function that accepts `(component_id, component_name, field_layouts, flags)` from Python's `ComponentMeta.__new__()` and stores metadata in a global `TypeRegistry` backed by `parking_lot::RwLock<HashMap<u32, ComponentTypeInfo>>`. The core logic -- tuple-to-struct conversion, field validation, size computation, and registry insertion -- is functionally correct.

**Verdict**: PASS with 1 isolated bug, 4 code-quality issues, and 3 test-coverage gaps. No correctness defects in the hot path.

---

## Bug Findings

### BUG-1. `compute_component_size` reports byte-extent, not struct size (medium severity)

**Location**: `type_bridge.rs`, lines 76-82.

**Description**: The function computes `max(field.offset + type_code_size(field.type_code))`. This returns the offset of the last byte + 1 -- the **extent** of the declared fields -- but does not account for trailing struct padding required by alignment.

For example, a component with a single `u8` at offset 0 and a `u64` at offset 1 would report size = `max(0+1, 1+8)` = **9** bytes. The correct ABI-aligned size for that layout on x86-64 is **16** bytes (the `u64` at offset 1 is itself misaligned, but even with offset 8 for a properly aligned `u64`, the struct would need to be 16 bytes due to trailing alignment padding).

**Impact**: If the ECS system ever uses the `size` field for:
- raw memcpy of component data (e.g., `std::ptr::copy_nonoverlapping`),
- allocating tightly-packed SOA buffers, or
- serialization with strict alignment requirements,

then undersized allocations could lead to buffer over-reads or misaligned accesses.

**Mitigation**: The Python `ComponentMeta.__new__()` defines exact byte offsets, and the current ECS layout may never reach through this code path for those operations. If the `size` is only used for bookkeeping / display / validation, the computed extent is sufficient.

**Recommendation**: Either (a) rename to `compute_component_extent()` and document that it does NOT include trailing padding (callers must align up), or (b) add an explicit trailing-padding computation that rounds up to `max_field_alignment`:

```rust
fn compute_component_size(fields: &[FieldLayout]) -> usize {
    let extent = fields.iter()
        .map(|f| f.offset + type_code_size(&f.type_code))
        .max()
        .unwrap_or(0);
    // Round up to the largest field's alignment
    let max_align = fields.iter()
        .map(|f| type_code_align(&f.type_code))
        .max()
        .unwrap_or(1);
    (extent + max_align - 1) & !(max_align - 1)
}
```

---

## Code Quality Findings

### CQ-1. `FieldType` enum is dead code (low severity)

**Location**: `type_registry.rs`, lines 12-19.

The `FieldType` enum (`F32`, `I32`, `U8`, `String`, `Fixed16`, `Fixed32`) is defined but never referenced anywhere in the crate. `type_code_size()` uses string matching instead.

```rust
pub enum FieldType {
    F32, I32, U8, String, Fixed16, Fixed32,
}
```

**Impact**: Dead code that will earn a compiler warning (or lint violation) and could mislead a reader into thinking the registry uses typed field discriminators.

**Recommendation**: Either (a) remove the enum, or (b) refactor `type_code_size` to accept `FieldType` instead of `&str` and provide a `FromStr` impl. The string-based approach is more flexible for Python interop, so option (a) is probably the better call.

### CQ-2. No validation for `component_id == 0` (low severity)

**Location**: `type_bridge.rs`, lines 120-167.

The `type_register` function accepts any `u32` for `component_id`, including 0. Many ECS systems reserve 0 as an invalid/placeholder ID. Whether this is valid depends on the convention adopted by the Python side, but there is no guard.

**Impact**: If the Python side accidentally passes 0 as a component ID, the error will manifest as a silent overwrite or a hard-to-debug lookup failure rather than a clear rejection at the registration boundary.

**Recommendation**: Add an explicit `if component_id == 0` guard if the system reserves 0. If 0 is valid (e.g., for unit-testing or internal use), document this choice.

### CQ-3. `archetype_id` is unconditionally `None` at registration (low severity)

**Location**: `type_bridge.rs`, line 162. `type_registry.rs`, lines 55-64.

The `ComponentTypeInfo` struct has an `archetype_id: Option<ArchetypeId>` field. Every call to `type_register` sets it to `None`. The field is `pub` and has no setter, so the only way to assign it is a direct mutation of the struct after lookup.

**Impact**: The `archetype_id` semantic is confusing -- the field exists, it is documented ("The archetype this component is assigned to"), but it is always `None` after registration. Callers who use `TypeRegistry::get()` will always see `None`.

**Recommendation**: Either (a) remove the field from `ComponentTypeInfo` and let archetype assignment live in a separate mapping (cleaner separation of concerns), or (b) add a `TypeRegistry::assign_to_archetype(id, archetype_id)` method and document that archetype assignment is a separate, later step.

### CQ-4. Duplicate ID registration is silent (low severity)

**Location**: `type_bridge.rs`, line 165, delegating to `TypeRegistry::register` at `type_registry.rs`, line 83-85.

`TypeRegistry::register` calls `self.types.write().insert(info.id, info)`, which silently overwrites any existing entry with the same ID. There is no warning, log, or error.

```rust
pub fn register(&self, info: ComponentTypeInfo) {
    self.types.write().insert(info.id, info);
}
```

**Impact**: If the Python side double-registers a component ID (e.g., two component classes with the same numeric ID), the first registration is silently lost. The Python side may see inconsistent metadata depending on registration order.

**Recommendation**: Consider returning a `bool` (or `Result`) indicating whether an existing entry was overwritten, or at minimum document the overwrite semantics in the doc comment. The `type_register` function could optionally support a `strict: bool` parameter that rejects overwrites.

---

## Test Coverage Gaps

### GAP-1. No test for `type_register` validation error paths (high severity)

The `type_register` function has three validation checks that return `PyValueError`:
1. Empty `component_name` (line 128)
2. Empty field `name` at index (line 137)
3. Empty field `type_code` (line 142)

None of these paths are tested. The existing test `test_validate_field_layouts_via_local_equivalent` (lines 434-469) only tests the success path with valid inputs.

**Missing tests**:
- `type_register` with empty component_name -> PyValueError
- `type_register` with an empty field name -> PyValueError at the correct index
- `type_register` with an empty type_code -> PyValueError with the correct field name

Since these cannot be tested through PyO3 without a Python runtime, the recommendation is to factor the validation logic into a pure-Rust `fn validate_field_layouts(...) -> Result<...>` function that is tested directly, and then call it from `type_register`.

### GAP-2. No concurrent-access stress test for the global registry (medium severity)

The global registry uses `OnceLock<Arc<TypeRegistry>>` backed by `parking_lot::RwLock`. This is correct by construction, but there is no test that exercises concurrent reads and writes to the global registry under contention.

`test_global_registry_can_register` (lines 521-536) uses `global_registry()` but only performs a single sequential register + get.

**Missing**: A test that spawns N reader threads that call `TypeRegistry::get()` in a loop while a writer thread registers new types, verifying no panics, no deadlocks, and consistent reads (each read returns either the old value or the new value, never a torn/corrupted state).

### GAP-3. `test_global_registry_can_register` mutates shared global state (low severity)

**Location**: `type_bridge.rs`, lines 521-536.

This test calls `global_registry()` (the static `OnceLock` singleton) and registers component ID 9999 into it. The test comment acknowledges: "This component might persist across tests -- that's fine for a smoke test."

Since `cargo test` runs unit tests in the same process by default, the mutation of the global registry persists for the lifetime of the test process. Any future test that accesses the global registry will see ID 9999 as a leftover. This is a latent test-isolation issue.

**Recommendation**: Either (a) remove the test (it adds no value beyond what `test_registry` tests already cover), or (b) mark it with `#[ignore]` and add a comment explaining it is an opt-in manual smoke test, or (c) make it a separate integration test binary that runs in its own process.

---

## Strengths

1. **Clean module boundary.** The separation between `type_bridge.rs` (PyO3 concerns) and `type_registry.rs` (pure data structures + locking) is well-delineated. The bridge file has no business logic beyond conversion and validation.

2. **Thread-safety is correct.** `OnceLock<Arc<TypeRegistry>>` guarantees safe lazy initialization. `parking_lot::RwLock` provides writer-priority access without spinning. Readers never block each other. No unsafe code.

3. **Validation exists at the correct level.** Field name and type_code emptiness checks happen during tuple-to-`FieldLayout` conversion, before `ComponentTypeInfo` construction. The error messages include the field index and name, which aids Python-side debugging.

4. **Inline tests are comprehensive for the computation helpers.** `type_code_size` is tested for all known type codes plus unknown/corner cases. `compute_component_size` is tested for empty, single-field, multi-field, non-contiguous, and mixed-type layouts.

5. **Field layout order is preserved.** `test_field_layout_order_preserved` explicitly verifies that fields are stored in insertion order, not sorted by offset or name. This is important because Python `ComponentMeta` may define fields in declaration order.

6. **No correctness bugs in the hot path.** The `type_register` function correctly maps Python tuples to Rust structs, applies validation before mutation, and stores data under the correct key. The three-way handshake (Python -> PyO3 -> TypeRegistry) is sound.

---

## Test Inventory

### Whitebox: frame_graph::type_bridge::tests (16 tests)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_type_code_size_known_types` | All known type codes map to correct byte sizes |
| 2 | `test_type_code_size_unknown_defaults_to_four` | Unknown codes default to 4 bytes |
| 3 | `test_compute_size_empty_fields_is_zero` | Zero fields = size 0 |
| 4 | `test_compute_size_single_field` | Single f32 at offset 0 = size 4 |
| 5 | `test_compute_size_multiple_fields` | Three f32s at offsets 0, 4, 8 = size 12 |
| 6 | `test_compute_size_uses_max_offset_plus_type_size` | Large offset + small type = max extent |
| 7 | `test_compute_size_non_contiguous_offsets` | Gap between fields computed correctly |
| 8 | `test_register_component_via_local_registry` | Register + get round-trip through local registry |
| 9 | `test_register_component_with_flags` | Flags field preserved through register/get |
| 10 | `test_register_component_with_no_fields` | Zero-field (tag) component registers cleanly |
| 11 | `test_register_multiple_components_unique_ids` | Bulk register of 5 IDs, len() and ids() correct |
| 12 | `test_register_overwrites_existing` | Second register with same ID overwrites |
| 13 | `test_validate_field_layouts_via_local_equivalent` | 3-field Vec3 round-trip through local registry |
| 14 | `test_field_layout_order_preserved` | Fields stored in insertion order (z, y, x) |
| 15 | `test_global_registry_is_some` | Global registry initializes without panic |
| 16 | `test_global_registry_can_register` | Global registry accept + retrieve (mutates shared state) |

### Whitebox: type_registry::tests (11 tests)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_new_registry_empty` | New registry is empty, len == 0 |
| 2 | `test_register_get_roundtrip` | Register + get by ID |
| 3 | `test_get_missing_returns_none` | Unknown ID returns None |
| 4 | `test_register_overwrites` | Same ID, second call overwrites |
| 5 | `test_type_list` | type_list() returns all registered (name, id, size) tuples |
| 6 | `test_archetype_for_same_set` | Same IDs, different order = same ArchetypeId |
| 7 | `test_archetype_for_different_sets` | Different ID sets = different ArchetypeId |
| 8 | `test_len_and_is_empty` | len() and is_empty() track state correctly |
| 9 | `test_contains` | contains() returns correct bool |
| 10 | `test_ids_returns_registered_ids` | ids() returns all IDs |
| 11 | `test_archetype_id_from_u32` | ArchetypeId -> u32 round-trip, non-zero for non-empty set |

---

## Verdict

**PASS** -- No release-blocking issues. The implementation is functionally correct with thread-safe global state, clean PyO3-to-Rust separation, and thorough unit coverage for the computation helpers.

**One bug to address before FINAL sign-off**:

| ID | Severity | File | Description |
|----|----------|------|-------------|
| BUG-1 | Medium | `type_bridge.rs:76-82` | `compute_component_size` does not account for trailing alignment padding; the returned value is a byte-extent, not an ABI-safe struct size. |

**Three coverage gaps to address before FINAL**:

| ID | Severity | Description |
|----|----------|-------------|
| GAP-1 | High | No test for any of the 3 `type_register` validation error paths (empty name, empty field name, empty type_code) |
| GAP-2 | Medium | No concurrent read/write stress test for the global registry |
| GAP-3 | Low | `test_global_registry_can_register` mutates the process-wide singleton; latent test-isolation risk |

**Recommendations for SENIOR_QA / FINAL sign-off**:
- Address BUG-1 by either renaming `compute_component_size` to `compute_component_extent` with documentation, or adding trailing-padding computation with alignment rounding
- Address GAP-1 by factoring validation into a pure-Rust function testable without PyO3
- GAP-2 and GAP-3 are hardening items -- file as follow-up tasks if concurrency guarantees need stronger evidence
- CQ-1 (dead `FieldType` enum) and CQ-4 (silent overwrite) are quick cleanup wins
