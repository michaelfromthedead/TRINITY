# T-FG-7.1 Type Channel -- Senior QA Sanity Report

**Reviewer**: Senior QA (Sanity Gate)
**Review date**: 2026-05-23
**Base review**: `workflows/SDLC/T-FG-7.1_type_channel_findings_junior.md` -- JUNIOR: PASS (1 medium bug, 4 code-quality, 3 coverage gaps)
**Source files audited**:
- `crates/renderer-backend/src/frame_graph/type_bridge.rs`
- `crates/renderer-backend/src/type_registry.rs`
- `crates/renderer-backend/src/component_store.rs` (cross-reference for `archetype_id`)
- `crates/renderer-backend/src/lib.rs` (re-exports)

---

## Verdict: JUNIOR FINDINGS -- 4 REAL / 4 OVERZEALOUS

The implementation is sound. The junior correctly identified dead code and test gaps. The "BUG-1" finding is overzealous -- the function behaves exactly as documented. Overall direction to FINAL: **PASS with advisory notes**.

---

## Finding-by-Finding Assessment

### BUG-1. `compute_component_size` reports byte-extent, not struct size

**Mark**: OVERZEALOUS

**Rationale**: The function's doc comment (lines 72-75 of `type_bridge.rs`) explicitly documents the formula: `max(field.offset + type_code_size(field.type_code))`. The module-level doc (lines 18-20) repeats it. The function does exactly what it says it does. It computes the byte-extent of the declared fields at their given offsets.

The junior's example (u8 at offset 0, u64 at offset 1 producing size=9 vs "correct" 16) is misleading: offset 1 for a u64 is already misaligned by ABI rules. Offsets come from Python's `ComponentMeta.__new__()`; if Python provides misaligned offsets, that is the upstream issue, not a bug in the extent computation.

The naming is imprecise ("size" rather than "extent"), which is a documentation concern. The junior's recommendation to rename the function or add alignment rounding is valid as a code-quality suggestion but does not rise to a bug. The junior also concedes: "If the size is only used for bookkeeping/display/validation, the computed extent is sufficient" -- which is the current state.

**Recommended action at FINAL**: Accept the renaming suggestion (`compute_component_extent`) as a low-priority cleanup, or add a doc note that callers must round up when allocating. Do not block on this.

---

### CQ-1. `FieldType` enum is dead code

**Mark**: REAL

**Rationale**: Confirmed. The enum at `type_registry.rs:12-19` (`F32, I32, U8, String, Fixed16, Fixed32`) is defined but never referenced anywhere in the crate. `grep -rn FieldType` across the entire `src/` directory returns only the definition line. The module is `pub mod type_registry` from `lib.rs`, so the enum is publicly visible, but internally it is dead weight -- it will produce a compiler warning on `#![warn(dead_code)]`.

The string-based matching in `type_code_size()` (`type_bridge.rs:58-70`) covers the same conceptual domain without using the enum type.

**Recommended action at FINAL**: Remove the enum. If external consumers rely on it, deprecate it first. This is a trivial cleanup.

---

### CQ-2. No validation for `component_id == 0`

**Mark**: OVERZEALOUS

**Rationale**: The junior acknowledges: "Whether this is valid depends on the convention adopted by the Python side." There is no evidence in the codebase that 0 is a reserved/invalid ID. The `ArchetypeId::from_component_ids` test (`type_registry.rs:261`) explicitly asserts that non-empty sets produce non-zero IDs, but this is an internal implementation detail of the hashing function, not a system-wide convention. Speculating about future conventions is the role of a design discussion, not a code-quality finding.

**Recommended action at FINAL**: No action needed. If a convention that reserves 0 is adopted in the future, add a guard at that point.

---

### CQ-3. `archetype_id` is unconditionally `None` at registration

**Mark**: REAL

**Rationale**: Cross-crate audit confirms the finding. Every construction of `ComponentTypeInfo` in the entire codebase sets `archetype_id: None`:

| File | Lines |
|------|-------|
| `type_bridge.rs` | 162, 316, 352, 368, 389, 410, 422, 462, 502, 529 |
| `type_registry.rs` | 142, 182 |
| `editor.rs` | 151, 159 |
| `renderer.rs` | 590, 600 |
| `component_store.rs` | 339, 348, 357, 366 |

No production path ever sets it to `Some(...)`. In `component_store.rs:279`, `archetype_id` is used as a function parameter name (different scope), not the `ComponentTypeInfo` field. The field has documentation "The archetype this component is assigned to" but the API provides no method to assign it -- callers must resort to direct struct mutation after lookup.

This is a genuine API design gap. The field exists and is documented, but the registration API never populates it and provides no setter.

**Recommended action at FINAL**: Add a `TypeRegistry::assign_to_archetype(id, archetype_id)` method, as the junior suggested. This is a quick, non-breaking addition.

---

### CQ-4. Duplicate ID registration is silent

**Mark**: OVERZEALOUS

**Rationale**: `HashMap::insert` semantics (last-write-wins) are a well-known design choice, not a defect. The overwrite behavior is explicitly tested in TWO places:
- `type_registry.rs:172-188` -- `test_register_overwrites`
- `type_bridge.rs:400-429` -- `test_register_overwrites_existing`

Both tests verify that the second registration replaces the first. This is intentional, tested, and documented (via tests as living documentation). Returning a `Result<bool>` from `register()` would be an API break with no demonstrated need.

**Recommended action at FINAL**: No action. The behavior is intentional. If the Python side needs duplicate detection, add it at the `type_register` level (in the bridge) with a separate key-exists check before calling `register()`.

---

### GAP-1. No test for `type_register` validation error paths

**Mark**: REAL

**Rationale**: The `type_register` function has three validation checks (lines 127-145 of `type_bridge.rs`) that return `PyValueError`:
1. Empty `component_name`
2. Empty field `name` at index
3. Empty field `type_code`

None of these paths have tests. The test `test_validate_field_layouts_via_local_equivalent` (line 433) only exercises the success path. Because these paths go through `#[pyfunction]`, they require a Python runtime to invoke -- standard `cargo test` cannot reach them.

The junior's recommendation to factor validation into a pure-Rust `fn validate_field_layouts(...) -> Result<...>` is the correct approach. This enables direct unit testing without PyO3 and follows the existing pattern of separating bridge logic (`type_bridge.rs`) from pure-domain logic (`type_registry.rs`).

This is the highest-priority actionable finding.

**Recommended action at FINAL**: Extract the three validation checks into a standalone pure-Rust function. Write three tests targeting it.

---

### GAP-2. No concurrent-access stress test for the global registry

**Mark**: OVERZEALOUS

**Rationale**: The global registry uses `OnceLock<Arc<TypeRegistry>>` backed by `parking_lot::RwLock`:
- `OnceLock` (stabilized in Rust 1.70) is a well-tested standard library primitive for lazy initialization.
- `parking_lot::RwLock` is the de-facto standard RwLock in the Rust ecosystem, proven across thousands of production deployments. It is specifically designed to prevent writer starvation.
- The access pattern is trivial: readers call `get()`, writers call `register()`. No lock is held across await points, no lock ordering exists, no re-entrancy.

A stress test for these primitives would test `parking_lot` and `std` internals, not application logic. This is not a meaningful coverage gap. The junior correctly notes the implementation is "correct by construction."

**Recommended action at FINAL**: No action. If the registry grows significantly more complex in the future (e.g., iterators that hold locks), add a concurrency test at that point.

---

### GAP-3. `test_global_registry_can_register` mutates shared global state

**Mark**: REAL

**Rationale**: The test at `type_bridge.rs:520-536` registers component ID 9999 into the process-wide global singleton. The test comment acknowledges the side effect. `cargo test` runs unit tests in parallel within the same process by default, so this mutation is theoretically observable by other tests.

However, the practical risk is very low: no other test in the crate reads from or depends on the global registry. The only peer test (`test_global_registry_is_some`) merely checks that initialization succeeds. The risk is latent -- it would only surface if a future test reads the global and happens to check for ID 9999.

The junior's recommendations (remove the test, mark it `#[ignore]`, or move it to an integration test binary) are all reasonable.

**Recommended action at FINAL**: Either remove the test (it adds negligible value beyond what local-registry tests already cover) or mark it `#[ignore]` with a comment. If a concurrency stress test is ever written in the future (GAP-2 notwithstanding), it should use a fresh `TypeRegistry`, not the global.

---

## Summary

| ID | Junior finding | Senior verdict |
|----|---------------|----------------|
| BUG-1 | `compute_component_size` is extent, not struct size (medium) | **OVERZEALOUS** -- behaves as documented; naming concern only |
| CQ-1 | `FieldType` enum is dead code (low) | **REAL** -- confirmed unreferenced in crate |
| CQ-2 | No validation for component_id == 0 (low) | **OVERZEALOUS** -- speculative convention; no evidence 0 is reserved |
| CQ-3 | archetype_id always None at registration (low) | **REAL** -- confirmed always-None across entire codebase |
| CQ-4 | Duplicate ID registration is silent (low) | **OVERZEALOUS** -- intentional, tested, documented behavior |
| GAP-1 | No test for validation error paths (high) | **REAL** -- highest-priority finding; three untested paths |
| GAP-2 | No concurrent stress test (medium) | **OVERZEALOUS** -- would test standard library, not app logic |
| GAP-3 | Global test mutates shared state (low) | **REAL** -- latent test-isolation risk, though low practical impact |

**REAL count**: 4 (CQ-1, CQ-3, GAP-1, GAP-3)
**OVERZEALOUS count**: 4 (BUG-1, CQ-2, CQ-4, GAP-2)

## Instructions for FINAL Gate

**Must address before sign-off**:
1. **GAP-1**: Factor validation into a pure-Rust function and add tests for the three error paths.
2. **CQ-3**: Add a `TypeRegistry::assign_to_archetype()` method so the `archetype_id` field is settable through the API.
3. **CQ-1**: Remove the dead `FieldType` enum.
4. **GAP-3**: Remove, ignore, or isolate `test_global_registry_can_register`.

**Advisory (no block)**:
- BUG-1: Rename `compute_component_size` to `compute_component_extent` as a documentation improvement.
- CQ-2, CQ-4, GAP-2: No action needed.
