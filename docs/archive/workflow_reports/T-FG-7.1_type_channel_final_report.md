# T-FG-7.1 Type Channel -- Senior QA Final Report

**Reviewer**: Senior QA (Final Gate)
**Review date**: 2026-05-23
**Base reviews**:
- JUNIOR: 8 findings (1 BUG, 4 CQ, 3 GAP)
- SANITY: 4 REAL / 4 OVERZEALOUS
**Source files audited**:
- `crates/renderer-backend/src/frame_graph/type_bridge.rs`
- `crates/renderer-backend/src/type_registry.rs`
- `crates/renderer-backend/src/component_store.rs`
- `crates/renderer-backend/src/editor.rs`
- `crates/renderer-backend/src/renderer.rs`
- `crates/renderer-backend/src/lib.rs`

---

## Final Verdict: CONDITIONAL PASS -- 4 findings unaddressed

The implementation is sound. All 409 unit tests pass. No new issues were introduced by the type-channel code. However, **none of the 4 REAL findings from the sanity gate have been addressed**. This report documents each finding with its current status and the specific edits required to close it.

---

## Sanity Finding Resolution Status

### MUST-ADDRESS Findings (from Sanity Gate)

#### GAP-1. No test for `type_register` validation error paths (HIGH)

**Status**: NOT ADDRESSED

**Current state**: The three validation checks at `type_bridge.rs:127-145` (empty component_name, empty field name, empty type_code) remain only reachable through `#[pyfunction]`, which requires a Python runtime. No pure-Rust extraction exists. The sole test `test_validate_field_layouts_via_local_equivalent` (line 433) only exercises the success path.

**Required fix**: Extract the three validation checks into a standalone pure-Rust function in `type_bridge.rs`:

```rust
/// Validate field layout tuples. Returns Ok(()) or a descriptive error string.
/// This pure-Rust function can be unit-tested without a Python runtime.
fn validate_field_layouts(
    component_name: &str,
    field_layouts: &[(String, String, usize)],
) -> Result<(), String> {
    if component_name.is_empty() {
        return Err("component_name must not be empty".into());
    }
    for (idx, (name, type_code, _)) in field_layouts.iter().enumerate() {
        if name.is_empty() {
            return Err(format!("field at index {idx} has an empty name"));
        }
        if type_code.is_empty() {
            return Err(format!("field '{}' has an empty type_code", name));
        }
    }
    Ok(())
}
```

Then call it from `type_register` and write three tests targeting it directly.

**Severity**: High. Three error paths are exercised by nothing except manual Python testing.

---

#### CQ-3. `archetype_id` is unconditionally `None` at registration (LOW)

**Status**: NOT ADDRESSED

**Current state**: All 20 construction sites of `ComponentTypeInfo` across the entire codebase still set `archetype_id: None`:

| File | Lines |
|------|-------|
| `type_bridge.rs` | 162, 316, 352, 368, 389, 410, 422, 462, 502, 529 |
| `type_registry.rs` | 142, 182 |
| `editor.rs` | 151, 159 |
| `renderer.rs` | 590, 600 |
| `component_store.rs` | 339, 348, 357, 366 |

No `assign_to_archetype` method exists on `TypeRegistry`.

**Required fix**: Add a method to `TypeRegistry` in `type_registry.rs`:

```rust
/// Assign a registered component type to an archetype.
/// Returns None if no component with the given id is registered.
pub fn assign_to_archetype(&self, component_id: u32, archetype_id: ArchetypeId) -> Option<()> {
    let mut types = self.types.write();
    if let Some(info) = types.get_mut(&component_id) {
        info.archetype_id = Some(archetype_id);
        Some(())
    } else {
        None
    }
}
```

**Severity**: Low. No production code currently depends on `archetype_id` being `Some(...)`. This is an API completeness gap.

---

#### CQ-1. `FieldType` enum is dead code (LOW)

**Status**: NOT ADDRESSED

**Current state**: The enum at `type_registry.rs:12-19` (`F32, I32, U8, String, Fixed16, Fixed32`) is still defined and exported via `pub mod type_registry`. Zero references exist outside the definition line. It will produce a compiler warning under `#![warn(dead_code)]`.

**Required fix**: Remove the enum entirely from `type_registry.rs:12-19`:

```rust
// Remove this entire block:
// pub enum FieldType {
//     F32,
//     I32,
//     U8,
//     String,
//     Fixed16,
//     Fixed32,
// }
```

If external consumers depend on it (unlikely given zero internal references), mark `#[deprecated]` first and remove in a follow-up.

**Severity**: Low. No functional impact; dead code removal only.

---

#### GAP-3. `test_global_registry_can_register` mutates shared global state (LOW)

**Status**: NOT ADDRESSED

**Current state**: The test at `type_bridge.rs:520-536` still registers component ID 9999 into the process-wide global singleton with no isolation mechanism. `cargo test` runs unit tests in parallel by default.

**Required fix**: Choose one:
1. **Remove** the test (it adds negligible value beyond local-registry tests).
2. **Mark `#[ignore]`** with a comment explaining why.
3. **Move to an integration test** that gets its own process.

**Severity**: Low. No other test currently reads the global, so the risk is latent.

---

### Advisory Findings (No Block)

#### BUG-1. `compute_component_size` naming (Advisory)

**Status**: NOT ADDRESSED (advisory only)

The function name `compute_component_size` is imprecise -- it computes byte-extent, not struct-size with ABI padding. The doc comment explicitly documents the formula, and all callers use the result for bookkeeping/display/validation, not allocation. Renaming to `compute_component_extent` would improve clarity but is not a blocker.

---

### OVERZEALOUS Findings (No Action Needed)

| ID | Finding | Reason |
|----|---------|--------|
| CQ-2 | No validation for component_id == 0 | Speculative convention; no evidence 0 is reserved |
| CQ-4 | Duplicate ID registration is silent | Intentional HashMap semantics; explicitly tested in two places |
| GAP-2 | No concurrent-access stress test | Would test `parking_lot` and `std::sync::OnceLock`, not app logic |

---

## Senior QA Supplemental Findings

No additional issues discovered beyond those identified by the junior and confirmed by sanity. The codebase is otherwise clean: the type channel's three-layer architecture (PyO3 bridge -> TypeRegistry -> ComponentStore) is well-separated, all public APIs have doc comments, and validation exists at the correct boundary.

### Final source-line audit (key functions)

| Function | File:Line | Assessment |
|----------|-----------|------------|
| `type_register` | `type_bridge.rs:120` | Correct. Three validation checks, then registration. |
| `type_code_size` | `type_bridge.rs:58` | Correct. Covers 12 type codes + unknown default. |
| `compute_component_size` | `type_bridge.rs:76` | Correct per doc. Naming imprecise (advisory). |
| `global_registry` | `type_bridge.rs:46` | Correct. OnceLock + Arc pattern, thread-safe. |
| `TypeRegistry` API | `type_registry.rs:75-129` | Complete. Register, get, len, is_empty, ids, contains, type_list, archetype_for. |
| `ComponentStore` | `component_store.rs:79-298` | Correct. spawn, despawn, read_field, write_field, query, column_slice. |
| `ArchetypeId` | `type_registry.rs:26-44` | Correct. Deterministic hash of sorted component IDs. |

---

## Final Action Items

- [ ] **GAP-1** (HIGH): Extract `validate_field_layouts` as a pure-Rust function + add three tests
- [ ] **CQ-3** (LOW): Add `TypeRegistry::assign_to_archetype()` method
- [ ] **CQ-1** (LOW): Remove dead `FieldType` enum
- [ ] **GAP-3** (LOW): Remove or isolate `test_global_registry_can_register`
- [ ] (Advisory) Rename `compute_component_size` to `compute_component_extent` as doc improvement

---

## Summary

| Dimension | Result |
|-----------|--------|
| **Unit tests** | 409/409 pass |
| **JUNIOR findings** | 8 total: 4 REAL, 4 OVERZEALOUS |
| **REAL findings addressed** | 0 of 4 |
| **Action items remaining** | 4 (1 HIGH, 3 LOW) |
| **Production correctness** | Sound |
| **Verdict** | CONDITIONAL PASS -- 4 items remain unaddressed |

**Gate decision**: CONDITIONS NOT YET MET. The implementation is correct and the code is sound, but the 4 REAL findings from the sanity gate must be resolved before sign-off. GAP-1 (high) is the highest priority -- validation error paths that are only exercisable through a Python runtime are a testing blind spot. The three LOW items (CQ-1, CQ-3, GAP-3) are trivial cleanups that should be completed in the same pass.
