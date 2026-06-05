# T-FG-5.5 Serial Fallback -- FIX Re-Review (Junior QA)

**File reviewed:** `crates/renderer-backend/src/frame_graph/mod.rs`
**Function:** `serial_fallback()` (lines 4615-4687)
**Tests:** 9 tests (lines 12095-12313)
**DEV scope:** FG-2 (insertion-based interleaving) + FG-4 (async_set dedup)
**Prior report:** `workflows/SDLC/T-FG-5.5_serial_fallback_findings_junior.md`
**SANITY report:** `workflows/SDLC/T-FG-5.5_serial_fallback_sanity_report.md`
**SENIOR report:** `workflows/SDLC/T-FG-5.5_serial_fallback_final_report.md`
**Date:** 2026-05-23
**Test count:** 383 passed, 0 failed (9 serial_fallback + 374 other)

---

## Severity Legend

| Label | Meaning |
|-------|---------|
| **CRITICAL** | Will produce wrong execution order, duplicate execution, or silent data corruption at runtime. Must fix before integration. |
| **HIGH** | Defects that cause incorrect behavior under specific conditions, or dead code posing as a complete feature. |
| **MEDIUM** | Violations of documented contract, missing validation, or design flaws that will cause bugs as the codebase evolves. |
| **LOW** | Code quality, test coverage gaps, style, or maintainability issues. |

---

## Verdict: CONDITIONAL PASS

FG-2 and FG-4 fixes are correct for the stated scenarios. All 9 serial_fallback tests pass. The new tests validate the specific failure modes identified in the prior review. However, the FG-2 fix has a known limitation (the PassIndex-monotonicity assumption) that is not tested or documented. This re-review finds the fixes acceptable for the merged scope, with one new LOW finding documenting the remaining boundary.

---

## Fix Verdict Audit

### FG-2 [CRITICAL -- FIXED] -- Slow path interleaving corrected

**Status:** VERIFIED FIXED

**Original bug (lines 4523-4540 prior code):** The slow path appended all deferred async passes at the end of the serial list, destroying their dependency-respecting interleaved positions. A graphics pass at position 2 could read stale data from an async pass at position 1 that was pushed to position 4.

**DEV fix (lines 4654-4684):** Three-part insertion strategy:
1. Collect non-async passes from `order` preserving their relative sequence (lines 4655-4659).
2. Collect deferred async passes from `async_set` (not `async_passes`) and sort by PassIndex (lines 4664-4673).
3. For each deferred pass, insert it before the first non-async pass with a higher PassIndex (lines 4678-4684).

**Evidence of correctness -- example from FG-2:**
- Input: `order=[0(gfx), 2(gfx), 4(gfx)]`, `async_set={1, 3}`
- Step 1: `serial = [0, 2, 4]`
- Step 2: `deferred = [1, 3]` (sorted by PassIndex)
- Step 3: Insert 1 before first idx.0 > 1 (position 1, value PassIndex(2)) = `[0, 1, 2, 4]`
- Step 4: Insert 3 before first idx.0 > 3 (position 3, value PassIndex(4)) = `[0, 1, 2, 3, 4]`
- Result: `[0, 1, 2, 3, 4]` -- correct interleaving

**New test:** `serial_fallback_slow_path_interleaves_async_passes_by_index` (lines 12229-12254) directly validates this exact scenario and asserts the correct output.

**Stress test:** `serial_fallback_slow_path_many_interleaved_passes` (lines 12289-12313) validates with 4 async passes at scattered positions [1, 4, 6, 8] and 6 non-async passes [0, 2, 3, 5, 7, 9], verifying the insertion handles async passes at beginning, middle, and end positions.

**Known limitation (unresolved):** The fix assumes PassIndex monotonicity is a valid proxy for topological ordering (line 4670-4672). This holds when passes are created and indexed in dependency order, but can fail when:
- Reverse dependency edges exist (e.g., pass 4 depends on pass 0 -- topological sort places 4 before 0, but sort-by-PassIndex places 0 first).
- Async passes have inter-dependencies that are not reflected by PassIndex order.

This limitation is shared with the original topological_sort tiebreaker (line 1959 sorts zero-in-degree sets by PassIndex). In practice, pass creation order typically follows dependency chains, so this heuristic is stable. But it should be documented as an assumption. See new finding FG-10.

---

### FG-4 [HIGH -- FIXED] -- Duplicate async entries deduplicated

**Status:** VERIFIED FIXED

**Original bug:** `deferred` iterated over `async_passes` (the raw `Vec<(PassIndex, String)>`), so a duplicate (PassIndex(1), "compute") and (PassIndex(1), "copy") would both pass the filter and be pushed into `deferred`, producing duplicate entries in the output. At runtime, the same GPU pass would be recorded and submitted twice.

**DEV fix (line 4664):** Changed the iterator source from `async_passes` to `async_set`:
```rust
let mut deferred: Vec<PassIndex> = async_set
    .iter()
    .copied()
    .filter(|idx| !order.contains(idx))
    .collect();
```

Since `async_set` is a `HashSet<PassIndex>`, duplicate entries from `async_passes` are automatically deduplicated before the deferred collection phase. This is a one-line change in the correct location.

**New test:** `serial_fallback_slow_path_deduplicates_duplicate_async_entries` (lines 12261-12283) directly validates:
- Input: `async_passes = [(1, "compute"), (1, "compute"), (3, "copy")]`, `order = [0, 2]`
- Expected: `[0, 1, 2, 3]` -- PassIndex(1) appears once
- The test asserts the output matches the deduplicated expectation.

No remaining concerns with this fix.

---

## Remaining Open Findings (from prior review)

These findings from the prior Junior/Senior reviews were NOT in scope for the DEV fix and remain open:

| Finding | Severity | Status | Notes |
|---------|----------|--------|-------|
| FG-1 | CRITICAL (prior) / MEDIUM (SENIOR downgrade) | OPEN | `serial_fallback` has zero production callers. `compile_with_config` (line 3306) builds the full async schedule but never calls `serial_fallback` to produce a serial fallback plan when `ASYNC_COMPUTE` is unavailable. The function is a structurally complete implementation with no caller. |
| FG-3 | HIGH | OPEN | `_passes: &[IrPass]` at line 4616 is still unused (leading underscore). Docstring claims it is "used for index bounds" but no bounds checking exists. Still a lying API contract. |
| FG-5 | HIGH | OPEN | No type-level distinction between full topo sort and graphics queue for `order` parameter. The fast-path/slow-path split depends on what `order` contains, but there is no compile-time guarantee. |
| FG-6 | MEDIUM | OPEN | `async_set` HashSet construction is still copy-pasted at lines 4446 (build_async_plan -- inside `AsyncExecutionPlan`), 4626 (serial_fallback), and 4725 (compute_sync_points). No `async_pass_set()` helper extracted. |
| FG-7 | MEDIUM | OPEN | `is_async_pass()` at line 4576 performs O(n) linear scan. `serial_fallback` still builds a HashSet instead of calling it (correctly -- they serve different performance regimes, but the maintenance inconsistency remains). |
| FG-8 | MEDIUM | OPEN | The original slow-path test `serial_fallback_merges_async_passes_when_order_excludes_them` (lines 12134-12170) still uses sequential indices `[4, 5]` for async and `[0, 1, 2, 3]` for non-async, where appending works by accident. The new FG-2 fix test supersedes this, but the old test is still present and still has weak validation. |
| FG-9 | MEDIUM | OPEN | `build_secondary_timeline` (lines 12502-12508) still has dead `_passes` and `_edges` parameters and produces an unsorted timeline. Still called from `compile_with_config` at line 3354. |

---

## New Finding FG-10 [LOW] -- PassIndex monotonicity assumption is undocumented

**Location:** Lines 4670-4673.

```rust
// 3. Sort by PassIndex for a deterministic, dependency-respecting order.
//    topological_sort assigns indices to reflect dependency order, so
//    sorting by PassIndex is a valid topological ordering heuristic.
deferred.sort_by_key(|idx| idx.0);
```

**Issue:** The comment asserts that "sorting by PassIndex is a valid topological ordering heuristic" but this is not universally true. `topological_sort` (line 1900) uses BFS with PassIndex as a tiebreaker within each zero-in-degree set (line 1959), but the overall topological order is determined by dependency edges. When reverse dependencies exist (a high-index pass depends on a low-index pass), the topological order can differ from PassIndex order, and the insertion-based merge will produce incorrect results.

**Demonstration of the failure case:**
- Passes created in order: P0(compute-async), P1(gfx), P2(gfx), P3(gfx), P4(gfx)
- Dependencies: P4 built by user code as a consumer of P0's output, with edge P4->P0 (P4 reads P0's output, so P4 depends on P0 completing) -- this is a RAW dependency, which creates a constraint: P0 must run before P4.
- Wait, that's not a reverse dependency. Let me reconsider.

Actually, the counterexample requires a dependency edge where a pass with a HIGHER index must run BEFORE a pass with a LOWER index. For example:
- P3 depends on P0 (edge P3->P0), but P3 and P0 both start with in-degree 0 from the edges array.
- Wait, P3->P0 means P0 has in-degree 1 (from P3) and P3 has in-degree 0.
- BFS: zero-in-degree = {P1, P2, P3, P4}, sort by PassIndex = [P1, P2, P3, P4]
- Process P1, P2, P3, P4 in order. When P4 is processed, we decrement P0's in-degree. P0 becomes zero-in-degree.
- Full order: [P1, P2, P3, P4, P0]

If P0 is async and the graphics queue is [P1, P2, P3, P4]:
- deferred = [P0], sorted = [P0]
- Insert P0 before first idx.0 > 0 = position 0
- Result: [P0, P1, P2, P3, P4]
- But topological order was [P1, P2, P3, P4, P0] (P4 must finish before P0 can start due to the dependency)
- Wait, P4->P0 means P4 writes something P0 reads, so P0 should run AFTER P4. But PassIndex(P0)=0 < PassIndex(P4)=4, so the insertion places P0 at position 0, before P4. WRONG.

Actually wait, P4->P0 could be a RAW edge: P4 reads resource R produced by P0. In that case, P0 must run before P4, and dependency is P0->P4 (the topological sort would put P0 before P4). I'm conflating edge direction semantics.

Let me reconsider: In the frame graph's `build_dag` function, edges represent resource dependencies. An edge P4->P0 with RAW type means P4 reads a resource written by P0. So P0 writes, P4 reads. P0 must come first. The topological sort would put P0 before P4.

But if PassIndex(P4) < PassIndex(P0), then P4 was created before P0 (since indices are sequential). A RAW edge P4->P0 means P4 reads what P0 writes -- but if P0 is created later, this seems impossible. Unless P0 is a late-inserted pass.

In practice, the compile pipeline creates passes in a forward pass, so PassIndex ordering is topological ordering. The BFS tiebreaker by PassIndex reinforces this.

Still, the assumption is not formally guaranteed and should be documented. This is a LOW finding.

**Action:** Add a doc-comment note on `serial_fallback` or the slow path that the insertion strategy relies on PassIndex monotonicity as a proxy for topological order, and that it may produce incorrect interleaving when the topological sort order diverges significantly from pass creation order (e.g., with late-inserted or reordered passes).

---

## Test Coverage Assessment (Post-Fix)

| Category | Tests | Coverage Notes |
|----------|-------|----------------|
| Fast path (all async in order) | 4 tests | `empty_async_passes`, `preserves_all_passes`, `with_mixed_passes`, `no_async_passes_bypasses` |
| Fast path (producer-consumer chain) | 1 test | `handles_producer_consumer_chain` -- validates 3-pass chain with async in middle |
| Slow path (order excludes async) | 1 test | `merges_async_passes_when_order_excludes_them` -- weak, sequential indices |
| **FG-2: slow path interleaved indices [NEW]** | 1 test | `slow_path_interleaves_async_passes_by_index` -- directly validates the fix |
| **FG-4: duplicate async dedup [NEW]** | 1 test | `slow_path_deduplicates_duplicate_async_entries` -- directly validates the fix |
| **FG-2 stress: many interleaved passes [NEW]** | 1 test | `slow_path_many_interleaved_passes` -- 10-pass range with scattered async |
| Empty order | 0 tests | Still not tested |
| Order with only async passes | 0 tests | Still not tested |
| Async-to-async inter-dependencies | 0 tests | Still not tested |
| Reverse dependency (PassIndex divergence) | 0 tests | NEW gap from finding FG-10 |

---

## Summary

| Metric | Value |
|--------|-------|
| Tests prior to fix | 6 serial_fallback tests |
| Tests after fix | 9 serial_fallback tests (+3 new) |
| Total tests passing | 383 (0 failures) |
| Findings in scope for DEV | 2 (FG-2, FG-4) |
| Findings VERIFIED FIXED | 2 (FG-2, FG-4) |
| Open findings (out of scope) | 7 (FG-1, FG-3, FG-5, FG-6, FG-7, FG-8, FG-9) |
| New finding in re-review | 1 (FG-10, LOW) |

**Bottom line:** The FG-2 and FG-4 fixes are correct and properly validated by 3 new tests. The insertion-based interleaving strategy correctly handles the case where async passes must be merged back at their dependency-respecting positions, and the async_set-based deferred collection correctly deduplicates duplicate async entries. The one remaining concern (FG-10, PassIndex monotonicity assumption) is LOW severity and acceptable for the current stage. The function remains dead code (no production callers in `compile_with_config`), which means all 7 open findings are non-blocking for this merge but must be tracked before T-FG-5.5 can be marked complete.

**Verdict from this re-review: CONDITIONAL PASS** -- FG-2 and FG-4 fixes are accepted. Open findings (FG-1, FG-3, FG-5, FG-6, FG-7, FG-8, FG-9, FG-10) should be triaged as follow-up work.
