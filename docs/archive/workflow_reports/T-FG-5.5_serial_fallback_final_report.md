# T-FG-5.5 Serial Fallback -- Senior QA Final Report

**Review file:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`
**Function:** `serial_fallback()` (lines 4495-4545)
**Tests:** 6 tests (lines 11880-12005)
**SANITY source:** `workflows/SDLC/T-FG-5.5_serial_fallback_sanity_report.md`
**Date:** 2026-05-23
**Role:** SENIOR_QA_FINAL -- independent verification of SANITY findings, root-cause synthesis, and go/no-go decision.

---

## Verdict: FIX

`serial_fallback()` must be fixed before it is safe to integrate into any production dispatch path. The fast path (the documented use case -- `order` is the full topological sort) is correct, but the slow path contains an algorithmic bug that will produce silent data corruption, the `_passes` parameter is a lying API contract, and duplicate async-pass entries produce duplicate submission. The function is currently dead code (no production callers), which gives the team a safe window to fix these issues before integration.

**The function should NOT be rewritten from scratch** -- the fast-path logic is correct and already tested. The slow path repair eliminates the broken "filter then append" approach and replaces it with a merge-template strategy using the original topological order. The remaining issues are well-scoped.

---

## SANITY Verdict Audit

I have independently verified each of the SANITY reviewer's verdicts against the source code. All 8 verdicts are correct.

### Upheld Verdicts

| Finding | SANITY Verdict | Final Verdict | Rationale |
|---------|---------------|---------------|-----------|
| FG-1 (CRITICAL claimed) | OVERZEALOUS | UPHELD | The function is dead code, but calling this CRITICAL misdiagnoses structure as impact. `self.order` at line 3238 stores the full topological order including all non-dead passes (both graphics and async-eligible). An executor that needs serial dispatch can iterate `self.order` directly -- no flattening needed. The missing integration is a completeness gap (MEDIUM), not a runtime defect. |
| FG-2 (CRITICAL claimed) | REAL, HIGH | UPHELD, HIGH | Algorithmic bug confirmed. The slow path at lines 4536-4542 appends deferred async passes at the end, destroying their dependency-respecting interleaved positions. In the example [0(gfx), 1(async), 2(gfx)] where pass 2 reads pass 1's output, the output [0, 2, 1] causes pass 2 to read stale data. Currently dormant because no production caller reaches the slow path. Severity HIGH is correct -- real bug, unreachable in current code. |
| FG-3 (HIGH) | REAL, HIGH | UPHELD, HIGH | Confirmed. `_passes: &[IrPass]` at line 4496 is never read in the function body (leading underscore confirms suppression). The docstring at line 4472 claims it is "used for index bounds" but no bounds checking exists. This is a documented API contract that is not upheld. HIGH is appropriate -- lying docstring will cause maintenance confusion and hides a potential downstream index-out-of-bounds crash vector. |
| FG-4 (HIGH) | REAL, HIGH | UPHELD, HIGH | Confirmed. `deferred` at line 4536 iterates over `async_passes` (the unsorted list with potential duplicates) rather than `async_set` (the deduplicated HashSet). When `async_passes` contains duplicate `(PassIndex, String)` entries, both pass the filter and produce duplicate `PassIndex` values in the output. This causes double GPU submission at execution time. HIGH is appropriate. |
| FG-5 (HIGH) | REAL, HIGH | UPHELD, HIGH | Confirmed. The function accepts `&[PassIndex]` but produces drastically different output depending on whether `order` is the full topological sort (fast path, correct) or a filtered subset like `graphics_queue` (slow path, buggy). No type-level or runtime contract enforcement. HIGH is appropriate -- real API footgun. |
| FG-6 (MEDIUM) | REAL, MEDIUM | UPHELD, MEDIUM | Confirmed. Same `async_passes.iter().map(|(idx, _)| *idx).collect()` pattern appears at lines 4424, 4506, and 4583. If the async_passes representation changes, all three drift. MEDIUM is appropriate. |
| FG-7 (MEDIUM) | OVERZEALOUS | UPHELD | Confirmed. `is_async_pass()` at line 4456 performs O(n) linear scan. `serial_fallback` needs O(1) bulk HashSet construction for its hot path. These are different operations for different performance regimes. The proper unification is FG-6's proposed batch helper, not forcing a linear scan into a set-build. The observation that the APIs should be consistent is fair, but not a defect. |
| FG-8 (MEDIUM) | REAL, MEDIUM | UPHELD, MEDIUM | Confirmed. The sole slow-path test at line 11919 uses sequential indices `[4, 5]` for async passes and `[0, 1, 2, 3]` for non-async -- appending-at-end happens to produce the correct result by accident. No test validates where merged async passes land relative to non-async passes with interleaved indices. MEDIUM is appropriate. |

### Additional NEW Finding (SANITY passed over this)

**FG-9 [MEDIUM -- NEW]: `build_secondary_timeline` replicates the dead-parameter pattern AND produces an unordered timeline.**

Location: Line 12150-12156.

```rust
fn build_secondary_timeline(
    async_passes: &[(PassIndex, String)],
    _passes: &[IrPass],       // <-- unused, same pattern as FG-3
    _edges: &[IrEdge],        // <-- unused, same pattern as FG-3
) -> Vec<PassIndex> {
    async_passes.iter().map(|(idx, _)| *idx).collect()
}
```

This function:
1. Accepts `_passes` and `_edges` that are never read (identical dead-parameter pattern to FG-3).
2. Produces an async timeline in whatever order `async_passes` entries appear -- NOT a topologically sorted order. If async passes have inter-dependencies, this timeline would produce incorrect execution order on the async compute queue.

While this function is technically out of scope for T-FG-5.5, it shares the same incomplete-integration pattern as `serial_fallback`. The fix for FG-6 (extracting an `async_pass_set()` helper) should also cover this function's `async_set` construction to prevent drift across all four sites (line 4424, 4506, 4583, 12155).

---

## Root Cause Synthesis

The 6 REAL findings trace to two root causes:

### Root Cause A: Incomplete integration (FG-1, FG-3, FG-9)

The `_passes` dead-parameter pattern appears in `serial_fallback` (line 4496), `generate_barriers` (line 2636), and `build_secondary_timeline` (line 12152-12153). Someone added these parameters with the intention of adding index-bounds validation but never finished the work. The docstrings claim the validation exists. This suggests the entire async-compute integration (Phase 5) was implemented as a skeleton and the validation/safety layer was deferred.

**Fix:** Either add the promised validation (verify `PassIndex.0 < passes.len()` on output, return `Result`) or remove the parameters. The choice affects the `pub` API -- removing is a breaking change; adding validation is the non-breaking better option.

### Root Cause B: The slow path uses the wrong merge strategy (FG-2, FG-4, FG-5, FG-8)

The slow path's "filter then append" approach at lines 4526-4542 is fundamentally incorrect. It assumes that if an async pass is not in `order`, it has no ordering constraints relative to passes that ARE in `order`. This assumption is false -- async passes were at interleaved positions in the topological order precisely because of dependency edges. FG-4 (duplicate output) is a sub-case of this: the wrong data structure (`async_passes` instead of `async_set`) is used for the wrong stage.

**Fix Recommendation:** Replace the slow path entirely with a merge-template strategy:

```rust
// Correct approach: use the FULL topological order as the merge template.
// Insert only the async passes that are missing from `order`.
// Keep the deterministic order by walking the merge template.
let template = get_full_topological_order(); // must be available
let mut result: Vec<PassIndex> = Vec::with_capacity(template.len());
for &idx in &template {
    if order.contains(&idx) || async_set.contains(&idx) {
        result.push(idx);
    }
}
```

Better still: eliminate the `order` parameter entirely and accept the full topological order only. This makes the slow path unreachable by construction (fixing FG-5) and eliminates the complexity of two code paths.

---

## Coverage Assessment

| Category | Current | Target | Status |
|----------|---------|--------|--------|
| Fast path (all async in order) | 3 tests | 2 tests | SUFFICIENT |
| Fast path (empty async_passes) | 2 tests | 1 test | SUFFICIENT |
| Slow path (order excludes async) | 1 test | 4+ tests | GAP |
| Slow path with interleaved indices | 0 tests | 2 tests | GAP -- blocks FG-2 fix |
| Duplicate async_passes | 0 tests | 1 test | GAP -- blocks FG-4 fix |
| Empty order | 0 tests | 1 test | GAP |
| Order with only async passes | 0 tests | 1 test | GAP |
| `_passes` validation | 0 tests | 1 test | GAP -- blocks FG-3 fix |
| Async passes with inter-dependencies | 0 tests | 1 test | GAP |

---

## Priority Remediation Plan

| Priority | Finding | Action | Effort | Depends On |
|----------|---------|--------|--------|------------|
| P0 | FG-2 | Fix slow path merge algorithm | 1-2h | -- |
| P0 | FG-4 | Iterate `async_set` instead of `async_passes` for deferred collection | 15min | -- |
| P1 | FG-3 | Add index bounds validation or remove `_passes` parameter | 30min-1h | P0 (test first) |
| P1 | FG-8 | Add slow-path tests with interleaved indices | 1-2h | P0 (tests validate the fix) |
| P2 | FG-5 | Strengthen API contract -- consider removing `order` parameter, always use full topo sort | 1h | P0 |
| P2 | FG-6 | Extract `async_pass_set()` helper, apply to all 3 sites (+ line 12155) | 15min | -- |
| P3 | FG-9 | Fix `build_secondary_timeline` dead parameters and add topological ordering | 30min | P2 |

**Estimated total fix effort:** 4-7 hours for all issues, or 2-3 hours for the P0 blockers (FG-2 + FG-4) if shipping under schedule pressure with the understanding the function is still dead code.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Function integrated before FG-2 is fixed | LOW (dead code, no production caller) | CRITICAL (silent data corruption on slow path) | Do not wire into `compile()` until P0 fixes land. Add compile-time check. |
| `_passes` removal breaks external callers | LOW (no external callers exist) | MEDIUM (API breaking change) | Remove only if no callers exist outside `mod.rs`. Currently safe. |
| Duplicate async passes in production | LOW (current `async_schedule` may not produce duplicates) | HIGH (double submission) | Fix FG-4 as belt-and-suspenders regardless of current `async_schedule` behavior. |
| Slow path regression during fix | MEDIUM | HIGH | Ensure existing fast-path tests still pass. Add slow-path property-based tests. |
| Dead code never integrated | MEDIUM (design gap) | MEDIUM (T-FG-5.5 feature incomplete) | Track integration in `compile()` as a separate task. Not a `serial_fallback` fix. |

---

## Summary

| Metric | Value |
|--------|-------|
| SANITY findings audited | 8 |
| SANITY verdicts upheld | 8 / 8 |
| New findings (SENIOR_QA_FINAL) | 1 (FG-9) |
| Total REAL findings | 7 (FG-2, FG-3, FG-4, FG-5, FG-6, FG-8, FG-9) |
| P0 blockers | 2 (FG-2 algorithmic bug, FG-4 duplicate output) |
| Recommended verdict | FIX |
| Risk if shipped unfixed | HIGH -- slow path causes silent data corruption on interleaved passes |

**Bottom line:** The SANITY review was thorough and correct. The function can be fixed in 2-3 hours for P0 items (FG-2, FG-4) or 4-7 hours for all items including P1/P2 hardening. The function MUST NOT be wired into the production `compile()` path until FG-2 and FG-4 are resolved. The fast path is correct and well-tested; the slow path and API contract issues are scoped repairs, not a fundamental redesign.

**Verdict: FIX**
