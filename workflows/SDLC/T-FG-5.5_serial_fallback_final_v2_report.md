# T-FG-5.5 Serial Fallback -- SENIOR_QA_FINAL (2nd Cycle)

**SDLC Stage:** SENIOR_QA_FINAL -- 2nd cycle, independent verification
**Fix scope:** FG-2 (slow path interleaving) + FG-4 (duplicate async dedup)
**Source fix:** `crates/renderer-backend/src/frame_graph/mod.rs` -- `serial_fallback()` lines 4706-4778
**New tests:** 3 tests, lines 12334-12417
**SANITY report:** `workflows/SDLC/T-FG-5.5_serial_fallback_fix_sanity_report.md`
**Junior re-review:** `workflows/SDLC/T-FG-5.5_serial_fallback_fix_findings_junior.md`
**Prior SENIOR report:** `workflows/SDLC/T-FG-5.5_serial_fallback_final_report.md`
**Date:** 2026-05-23

---

## Verdict: PASS

The FG-2 and FG-4 fixes are correct, well-tested, and independently verified against the source code. The 7 out-of-scope findings remain open and should be tracked for the next cycle, but none block this merge. FG-10 is formally disproven -- the finding is overzealous by construction.

---

## Fix Verification

### FG-2 [CRITICAL] -- Slow path interleaving: VERIFIED CORRECT

**Original bug (prior code lines 4523-4540):** The slow path appended all deferred async passes at the end of `serial`, destroying their dependency-respecting interleaved positions. A graphics pass at PassIndex 2 reading a resource written by an async pass at PassIndex 1 would receive stale data after reordering to [0, 2, 3, 1].

**Fix (lines 4732-4775):** Four-step insertion-based merge:

1. **Collect non-async from order** (lines 4746-4749): filter `order` to passes not in `async_set`, preserving their relative sequence.
2. **Collect deferred async from async_set** (lines 4755-4759): passes in `async_set` but absent from `order`. Uses `async_set` (HashSet), not `async_passes` (Vec). This is also the FG-4 fix.
3. **Sort deferred by PassIndex** (line 4764): deterministic ordering.
4. **Insert at dependency-respecting position** (lines 4769-4774): for each deferred pass `async_idx`, find the first non-async pass in `serial` with `PassIndex > async_idx.0` and insert before it. If no such pass exists, append at end.

**Independently verified correctness** -- the `build_dag` edge-direction invariant guarantees this works:

- `build_dag` creates edges only from `passes[i].index` to `passes[j].index` where `i < j` (lines 1850-1880, inner loop `b in (a+1)..`).
- Passes are enumerated in insertion order (line 1817), and PassIndex is assigned sequentially, so `passes[i].index.0 < passes[j].index.0` for all `i < j`.
- Therefore, every edge goes from a lower `PassIndex` to a higher `PassIndex` -- enforced both by construction and by the explicit test assertion at lines 9484-9490.
- This means sorting by `PassIndex` is **provably a valid topological order** (a linear extension of the partial order), not merely a heuristic.

**Three new tests validate the fix:**
- `serial_fallback_slow_path_interleaves_async_passes_by_index` (line 12334): direct scenario, async=[1,3], order=[0,2,4], expects [0,1,2,3,4].
- `serial_fallback_slow_path_many_interleaved_passes` (line 12394): 4 async passes at scattered positions [1,4,6,8] with 6 non-async passes [0,2,3,5,7,9], expects full sequential range [0..9].
- The existing legacy test `merges_async_passes_when_order_excludes_them` (line 12239) still passes and validates non-async ordering preservation.

**Status: VERIFIED CORRECT. No concerns.**

---

### FG-4 [HIGH] -- Duplicate async dedup: VERIFIED CORRECT

**Original bug (prior code line 4536):** `deferred` iterated over `async_passes` (raw `Vec<(PassIndex, String)>`), so duplicate `(PassIndex, String)` entries -- e.g., the same pass eligible for both "compute" and "copy" -- would both pass the filter and produce duplicate `PassIndex` values in the output, causing double GPU submission at execution time.

**Fix (line 4755):** Changed the iteration source from `async_passes` to `async_set`:

```rust
let mut deferred: Vec<PassIndex> = async_set       // <-- was: async_passes
    .iter()
    .copied()
    .filter(|idx| !order.contains(idx))
    .collect();
```

Since `async_set` is a `HashSet<PassIndex>`, duplicate entries are automatically eliminated before the deferred collection phase. This is a one-line change in exactly the right location.

**Dedicated test validates the fix:**
- `serial_fallback_slow_path_deduplicates_duplicate_async_entries` (line 12366): feeds `async_passes = [(1, "compute"), (1, "compute"), (3, "copy")]` with order `[0, 2]`, asserts output is `[0, 1, 2, 3]` (PassIndex 1 appears exactly once).

**Status: VERIFIED CORRECT. No concerns.**

---

## FG-10 Re-Assessment: CONFIRMED OVERZEALOUS

**Finding:** Junior argued that PassIndex monotonicity sorting (line 4764) "can fail" when topological order diverges from pass creation order.

**Independent judgment: REJECTED.** The finding is disproven by formal analysis of the codebase invariants:

1. **`build_dag` invariant (lines 1850-1880):** Every edge is constructed from `passes[i].index` to `passes[j].index` where `i < j` in insertion order. Since passes are created with sequentially incrementing PassIndex values, this means `from.0 < to.0` for every edge.

2. **Test enforcement (lines 9484-9490):** The invariant is explicitly tested:
   ```rust
   for edge in &edges {
       assert!(edge.from.0 < edge.to.0, ...);
   }
   ```

3. **Consequences for topological_sort:** The BFS tiebreaker (line 1959, `temp.sort()`) sorts zero-in-degree sets by PassIndex, which produces a valid topological order because all edges go from lower to higher PassIndex.

4. **Consequences for serial_fallback:** Since the full PassIndex order is itself a linear extension of the partial order, sorting deferred async passes by PassIndex produces a valid topological interleaving. This is not a heuristic -- it is provably correct.

**The Junior's own analysis disproves the finding.** After an extensive walkthrough, the Junior concluded "in practice, PassIndex ordering is topological ordering" -- which is true, and provably so under the `build_dag` contract.

**Potential residual (dismissed):** The Junior worried about "external edges" that violate the invariant. The `topological_sort` docstring warns these could come from external sources, but `serial_fallback`'s PassIndex sorting is not the tool to handle them -- the entire DAG would be invalid if `build_dag`'s contract is violated, and that is a caller bug.

| Criterion | Assessment |
|-----------|-----------|
| Describes a real bug | No -- provably impossible under documented invariants |
| Actionable remediation | None needed -- the assumption is guaranteed by construction |
| Would improve code if addressed | No -- formally verifiable, no counterexample exists |
| **Disposition** | **OVERZEALOUS -- REJECTED** |

---

## Out-of-Scope Findings Audit (7 remaining)

| Finding | Severity | Status | Notes from independent review |
|---------|----------|--------|-------------------------------|
| FG-1 | MEDIUM | OPEN | `serial_fallback` has zero production callers. `compile_with_config` builds the full async schedule at line ~3354 but never calls `serial_fallback`. The compile output is already correct for serial dispatch (full topological order in `self.order`), so this is a completeness gap, not a runtime defect. Must be resolved before T-FG-5.5 ships. |
| FG-3 | HIGH | OPEN | `_passes: &[IrPass]` at line 4707 is still unused (leading underscore suppresses the warning). Docstring at line 4683 claims it is "used for index bounds" but no bounds checking exists. Lying API contract -- should either implement validation or remove the parameter. |
| FG-5 | HIGH | OPEN | No type-level distinction between full topological sort and graphics queue for the `order: &[PassIndex]` parameter. The fast path (all async in order) vs. slow path (some async excluded) produces drastically different behavior with no compile-time guard. |
| FG-6 | MEDIUM | OPEN | `async_set` HashSet construction is copy-pasted at lines 4426 (`build_async_plan`), 4635 (`compute_sync_points` -- wait, that's the old `serial_fallback` location, let me correct: `build_async_plan` at 4426 / ...). Actually let me re-read: `build_async_plan` line 4635, `serial_fallback` line 4717, and `compute_sync_points` at line ~4725 (not visible in the current read). Three locations of the same `async_passes.iter().map(|(idx, _)| *idx).collect()` pattern. Extract helper to prevent drift. |
| FG-7 | MEDIUM | OPEN | `is_async_pass()` O(n) linear scan (line 4667) coexists with HashSet O(1) bulk lookup in `serial_fallback`. Different performance regimes, but the inconsistency is real. Naturally resolved by FG-6 (extracting a batch helper). |
| FG-8 | MEDIUM | OPEN | The legacy slow-path test at line 12239 (`merges_async_passes_when_order_excludes_them`) still uses sequential indices [4,5] for async and [0,1,2,3] for non-async, where append-at-end happens to work. Superseded by the FG-2 fix tests but still present. |
| FG-9 | MEDIUM | OPEN | `build_secondary_timeline` at line ~12502 replicates the dead-parameter pattern (`_passes`, `_edges` unused) and produces an unsorted timeline. Same incomplete-integration pattern from Root Cause A (prior SENIOR report). |

**All 7 findings remain OPEN. They are non-blocking for this merge cycle but must be triaged before T-FG-5.5 can ship.** None are in the scope of the DEV fix cycle (FG-2, FG-4).

---

## Risk Assessment (Post-Fix)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Function wired into `compile()` before P0 fixes | LOW (function still has no production caller) | CRITICAL (fixes are in place, but untested at integration) | Verify integration in `compile_with_config` before marking feature complete. |
| FG-2 insertion merge fails on non-standard edge patterns | NEGLIGIBLE (invariant is provable, not probabilistic) | N/A | Formal proof in this report. No edge case exists under `build_dag` contract. |
| Duplicate async passes in production | LOW (current `async_schedule` may not produce duplicates) | HIGH (belt-and-suspenders fix is applied) | FG-4 fix is already in place regardless. |
| Remaining 7 findings delay shipping | MEDIUM | MEDIUM | Triage separately. None block this merge. |

---

## Summary

| Metric | Value |
|--------|-------|
| Total fix findings in scope | 2 (FG-2, FG-4) |
| Fixes CORRECTLY VERIFIED | 2 / 2 |
| New finding from Junior re-review | 1 (FG-10) |
| FG-10 disposition | **REJECTED -- OVERZEALOUS** (provably impossible by invariant) |
| Remaining open findings | 7 (FG-1, FG-3, FG-5, FG-6, FG-7, FG-8, FG-9) |
| Open findings that block merge | **0** |
| Total test count | 383 passed (9 serial_fallback tests) |

**Formal proof of FG-2 correctness (independent):**
1. `build_dag` constructs edges only from `passes[i].index` to `passes[j].index` where `i < j` in insertion order (lines 1850-1880, inner loop `b in (a+1)..`).
2. Pass indices are sequential in insertion order, so every edge satisfies `from.0 < to.0`.
3. Therefore, the partial order defined by all edges is a sub-order of the total order defined by PassIndex.
4. Therefore, sorting by PassIndex produces a linear extension of the partial order -- a valid topological ordering.
5. The insertion-based merge at lines 4769-4774 places each deferred async pass at its PassIndex position in `serial`.
6. The result is a valid topological ordering containing all passes. QED.

FG-10 is not merely "overzealous" -- it is **formally wrong**. The concern it raises cannot manifest under the documented invariants of the codebase.

**Bottom line:** Both fixes are correct, the tests are adequate, and the out-of-scope findings do not block this merge. The function remains dead code, which provides a safe window to address the 7 remaining findings before integration.

**Action required:** The 7 remaining open findings must be assigned and tracked for the next cycle. The prioritized remediation plan from the first SENIOR_QA_FINAL report remains valid -- P0 items are fixed, P1-P3 items are the remaining work.

**FINAL VERDICT: PASS**
