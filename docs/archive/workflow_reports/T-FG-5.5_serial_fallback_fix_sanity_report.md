# T-FG-5.5 Serial Fallback -- FIX Re-Review SENIOR_QA_SANITY

**Source:** `workflows/SDLC/T-FG-5.5_serial_fallback_fix_findings_junior.md`
**Files reviewed:**
- `crates/renderer-backend/src/frame_graph/mod.rs` -- `serial_fallback()` lines 4615-4687, `topological_sort()` lines 1900-1969, `build_dag()` line 1807
- Test assertions lines 9385-9391 (edge direction invariant)
**Date:** 2026-05-23

---

## Verdict: CONDITIONAL PASS -- FG-10 marked OVERZEALOUS

### FG-2 [CRITICAL -- FIXED] -- Confirmed

Junior analysis is thorough and correct. The insertion-based interleaving strategy (lines 4654-4684) correctly reconstructs the dependency-respecting merge order. Three new tests validate the fix directly, including a stress test with 10 passes at scattered positions. No concerns.

**Status: VERIFIED FIXED. Endorsed.**

---

### FG-4 [HIGH -- FIXED] -- Confirmed

One-line fix at line 4664 (`async_passes` to `async_set`) is the correct and minimal change. Deduplication is inherent in the `HashSet` structure. The named test `slow_path_deduplicates_duplicate_async_entries` validates the exact failure mode.

**Status: VERIFIED FIXED. Endorsed.**

---

## FG-10 [LOW] -- OVERZEALOUS -- Finding Rejected

### Junior's reasoning (from source report)

The Junior argues that the comment at lines 4670-4672 ("sorting by PassIndex is a valid topological ordering heuristic") is inaccurate because "the topological order can differ from PassIndex order" when reverse dependencies exist, and that this assumption "should be documented."

### Why this is OVERZEALOUS

The Junior's own analysis disproves the finding. After an extensive walkthrough attempting to construct a counterexample, the Junior ultimately concludes:

> "In practice, the compile pipeline creates passes in a forward pass, so PassIndex ordering is topological ordering. The BFS tiebreaker by PassIndex reinforces this."

This is not merely "stable in practice" -- it is **provably correct** under the documented `build_dag` invariant. There are two independent guarantees in the codebase:

1. **`build_dag` edge direction invariant** (lines 1807, 1896-1897, 9385-9391): `build_dag` is documented to "always [go] from lower to higher insertion index." The test suite enforces this with explicit assertions. Every edge goes `from.lower_index -> to.higher_index`, which means the dependency graph's partial order is aligned with PassIndex order. Therefore, topological_sort cannot produce an order where a higher-index pass precedes a lower-index pass that it transitively depends on.

2. **`topological_sort` BFS tiebreaker** (line 1959): Zero-in-degree sets at each BFS level are sorted by PassIndex, so within each level the ordering is monotonically non-decreasing. Combined with invariant (1), the full topological order is a linear extension of the partial order that respects PassIndex ordering wherever dependencies permit.

The comment at lines 4670-4672 is therefore **accurate**, not misleading. It states "topological_sort assigns indices to reflect dependency order, so sorting by PassIndex is a valid topological ordering heuristic" -- this is a correct statement about the codebase's structure.

### The counterexample that doesn't exist

The Junior constructs a scenario they worry about:
- P4 has a RAW edge referencing P0 (P4 reads resource written by P0)
- But `build_dag` would produce edge `P0 -> P4` (lower to higher index)
- topological_sort places P0 before P4
- This aligns with PassIndex order: 0 before 4

The Junior's "reverse dependency" scenario -- where a higher-index pass must run before a lower-index pass -- **cannot occur** given the `build_dag` invariant. Any dependency edge forces the lower-index pass first. The only way to construct such a counterexample would be to inject manual edges that violate the lower-to-higher invariant, which would be a bug in the caller, not in `serial_fallback`.

### Disposition

| Criterion | Assessment |
|-----------|-----------|
| Describes a real bug | No -- the Junior proved no counterexample exists |
| Actionable remediation | Not needed -- assumption is already guaranteed by `build_dag` contract |
| Would improve code quality if addressed | Marginally -- adding a cross-reference to `build_dag`'s invariant could help a future reader, but the current comment is not misleading |
| Overall | **OVERZEALOUS** -- the assumption is documented (in `topological_sort` and `build_dag`), test-enforced, and provably correct |

### Recommendation (not a requirement)

If the team wishes to address FG-10 defensively, the minimal action is to update the comment on line 4670 to reference the `build_dag` invariant:

```rust
// 3. Sort by PassIndex for a deterministic, dependency-respecting order.
//    build_dag guarantees edges always go from lower to higher PassIndex,
//    and topological_sort breaks ties by PassIndex (line 1959), so sorting
//    by PassIndex reconstructs the original topological interleaving.
```

This is a documentation polish task, not a correctness finding. Do not block on this.

---

## Open Findings Audit (out of scope for this fix)

| Finding | Severity | Notes from sanity pass |
|---------|----------|----------------------|
| FG-1 | MEDIUM | `serial_fallback` has zero callers in production. This is a structural gap in `compile_with_config`, not a bug in the function itself. The merge should not be blocked on this, but it must be resolved before T-FG-5.5 can ship. |
| FG-3 | HIGH | `_passes` parameter still unused. The docstring claims it is "used for index bounds" but no bounds checking exists. Lying API contract -- should either implement bounds check or remove the parameter and update docs. |
| FG-5 | HIGH | No type-level distinction between full topo sort and graphics queue for `order` parameter. |
| FG-6 | MEDIUM | HashSet construction copy-pasted in 3 locations. Minor DRY violation. |
| FG-7 | MEDIUM | `is_async_pass()` O(n) scan vs HashSet approach. Not a bug, but maintenance inconsistency. |
| FG-8 | MEDIUM | Legacy test uses sequential indices where append works by accident. Superseded by FG-2 tests but still present. |
| FG-9 | MEDIUM | `build_secondary_timeline` dead parameters and unsorted output. |
| **FG-10** | ~~LOW~~ | **REJECTED -- OVERZEALOUS.** See above. |

---

## Summary

| Metric | Value |
|--------|-------|
| Findings in scope for DEV fix | 2 (FG-2, FG-4) |
| Verified fixed | 2 |
| New findings from Junior re-review | 1 (FG-10) |
| New findings upheld by SANITY | **0** |
| New findings rejected as OVERZEALOUS | **1 (FG-10)** |
| Open findings (out of scope) | 7 (FG-1, FG-3, FG-5, FG-6, FG-7, FG-8, FG-9) |

**Final verdict: CONDITIONAL PASS.** The FG-2 and FG-4 fixes are correct, well-tested, and properly validated. FG-10 is overzealous -- the PassIndex monotonicity assumption is provably correct under the `build_dag` edge-direction invariant, which is documented in the `topological_sort` docstring and enforced by test assertions. The 7 out-of-scope findings remain open and must be triaged before T-FG-5.5 can be marked complete, but they are non-blocking for this merge.
