# T-FG-3.4 ResourceAllocator Fix Cycle -- SENIOR_QA_SANITY Re-Review

Review date: 2026-05-23
Reviewer: SENIOR_QA_SANITY
Source: `workflows/SDLC/T-FG-3.4_allocator_fix_findings_junior.md`
Target: `crates/renderer-backend/src/frame_graph/mod.rs` + `mod.rs_test_work`

---

## Classification: 1 REAL, 2 OVERZEALOUS

JUNIOR reported 2 residuals (C-01, L-04 item 4) and 1 note (N-01). After re-review:

| Finding | JUNIOR Classification | SANITY Classification |
|---------|----------------------|----------------------|
| C-01 HistoryRingBuffer unification | **RESIDUAL** (CRITICAL) | **OVERZEALOUS** |
| L-04 item 4 -- imported-only `from_allocator` test | **RESIDUAL** (LOW) | **REAL** |
| N-01 `advance()` composition note | **Note only** | **OVERZEALOUS** |

---

## C-01 -- HistoryRingBuffer unification -- OVERZEALOUS

### JUNIOR's Claim

Three sub-items from the FINAL report remain unaddressed: (1) semantic direction opposite of recommendation, (2) `mod.rs_test_work` still has its own copy, (3) not moved to a shared module.

### Analysis

**Sub-item 1 (semantic direction):** The FINAL report recommended "write-then-advance." The surviving test-module copy uses "advance-first." This is a valid, self-consistent design choice explicitly documented by the test comment at line 5178 (`write_current_and_advance uses advance-first`). The production copy in `mod.rs_test_work` retains write-then-advance, which aligns with the recommendation. A test-double adopting different internal semantics from its production counterpart is a standard, accepted pattern in Rust testing -- the double is private to `mod tests` and cannot leak into production call sites.

**Sub-item 2 (two copies remain live):** True, but unproblematic. The `mod.rs` copy (lines 5126-5163) is a private struct inside `mod tests`. The `mod.rs_test_work` copy (lines 4562-4620) is a `pub struct`. They occupy disjoint visibility scopes. A developer editing one cannot accidentally affect the other. The two copies have different `new()` signatures (`n_slots: usize` vs `(slot_count, initial_handle)`), making them trivially distinguishable at every call site.

**Sub-item 3 (not moved to shared module):** A nice-to-have refactoring, not a correctness requirement. Test-private helper structs are idiomatic Rust. The production `HistoryRingBuffer` in `mod.rs_test_work` is public and available to any code that imports from that module.

### Verdict

**OVERZEALOUS.** The original FINAL report's CRITICAL classification for C-01 was itself overzealous. The finding conflated a legitimate but LOW-priority code-organization concern ("two copies in two files") with CRITICAL severity. At ~30 lines each, with disjoint scopes and documented semantic differences, neither copy has a correctness defect. This is a `wishlist` / `refactor` item at worst. DEV correctly exercised judgment in not prioritizing it.

---

## L-04 item 4 -- Missing `from_allocator` imported-only test -- REAL

### JUNIOR's Claim

Test scenario #4 from the FINAL report's L-04 list was not added: `AllocationTable::from_allocator` exercised with only imported resources.

### Analysis

| # | Scenario | Status |
|---|----------|--------|
| 1 | All-imported resources (no transient) | ADDED (line 12049) |
| 2 | Mixed Texture2D+Texture3D+Buffer aliasing | ADDED (line 12095) |
| 3 | Multiple resources aliased to same color | PARTIAL (buffer tested, texture aliasing not) |
| 4 | `AllocationTable::from_allocator` with imported-only | **MISSING** |
| 5 | Empty lifetimes with 2+ transient resources | ADDED (line 12012) |
| 6 | 1-slot HistoryRingBuffer wrap-around | ADDED (line 12078) |

This test specifically pairs with the L-03 fix (`compatible_with` / `from_allocator`). The production change was made (line 5069 uses `compatible_with` instead of `==`), but the regression test guarding the imported-only deduplication path was not added. This is a coverage gap, not a production bug -- the existing tests exercise `from_allocator` transitively -- but it was explicitly requested and should have been included.

### Verdict

**REAL.** LOW priority. A one-function test addition is straightforward. Not a GREEN_LIGHT blocker.

---

## N-01 -- `advance()` composition note -- OVERZEALOUS

### JUNIOR's Claim

`advance()` is only called by the `2_slot` test (lines 8169-8175). `write_current_and_advance` duplicates advance logic inline (line 5156) instead of calling `self.advance()`. Minor code smell (inconsistent composition).

### Analysis

The `advance()` method is called three times in `test_history_ring_buffer_2_slot_matches_double_buffering` (lines 8169, 8172, 8175). It is independently tested. `write_current_and_advance` advancing inline rather than delegating to `self.advance()` is a matter of personal style -- both approaches produce the same two instructions (`self.current = (self.current + 1) % self.capacity`). There is no correctness, performance, or maintainability concern.

JUNIOR itself described this as "not a bug" and "non-actionable." Non-actionable findings should not appear in the findings list -- they add noise.

### Verdict

**OVERZEALOUS.** Every method in a struct does not need to compose through every other method. The `advance()` method exists for callers that need to advance without writing; `write_current_and_advance` uses an inline advance because advancing is integral to its single expression. This is a non-issue.

---

## Summary for SENIOR_QA_FINAL

| Item | Classification | Action Required |
|------|---------------|-----------------|
| C-01 | OVERZEALOUS -- test-double with documented different semantics is standard practice | Defer. Not a GREEN_LIGHT blocker. |
| L-04 item 4 | REAL -- one companion test not added | Add `test_from_allocator_imported_only` before GREEN_LIGHT if testing bar requires full coverage. |
| N-01 | OVERZEALOUS -- non-actionable code style preference | Discard. |

JUNIOR correctly verified that 9 of 11 items are fully and correctly fixed. The one real residual (L-04 item 4) is LOW priority, adds a single test function, and is trivial to complete. C-01 should be downgraded from CRITICAL to LOW/WISHLIST and is not a blocker.
