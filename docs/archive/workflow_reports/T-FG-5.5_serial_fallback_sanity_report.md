# T-FG-5.5 Serial Fallback -- Senior QA Sanity Report

**Review file:** `crates/renderer-backend/src/frame_graph/mod.rs` -- `serial_fallback()` (lines 4493-4543)

**Junior findings source:** `workflows/SDLC/T-FG-5.5_serial_fallback_findings_junior.md`

**Date:** 2026-05-23

**Role:** Judicial. Each finding is marked REAL (bug/issue confirmed) or OVERZEALOUS (incorrect severity, wrong diagnosis, or not a genuine defect). One-line rationale per finding. No new findings.

---

## Verdict Summary

| Finding | Junior Severity | Senior Verdict | 
|---------|-----------------|----------------|
| FG-1    | CRITICAL        | OVERZEALOUS    |
| FG-2    | CRITICAL        | REAL           |
| FG-3    | HIGH            | REAL           |
| FG-4    | HIGH            | REAL           |
| FG-5    | HIGH            | REAL           |
| FG-6    | MEDIUM          | REAL           |
| FG-7    | MEDIUM          | OVERZEALOUS    |
| FG-8    | MEDIUM          | REAL           |

**Net:** 6 REAL, 2 OVERZEALOUS. 0 CRITICAL confirmed. 2 CRITICAL downgraded (FG-1 outright overzealous, FG-2 real but currently dormant).

---

## Detailed Verdicts

### Finding FG-1 [CRITIAL claimed] -- Zero production callers; function is dead code

**Verdict: OVERZEALOUS**

**Rationale:** The function is indeed never called outside tests, but it is **incorrect to call this CRITICAL** (i.e., "will cause wrong execution order, duplicate execution, or silent data corruption"). The `compile()` method at line 3238 stores the **full topological order** (all passes including async-eligible ones) in `self.order`. The executor can trivially dispatch from `self.order` for serial execution and ignore the `async_timeline` field -- no flattening needed, no data corruption risk. The missing integration hook means the `serial_fallback` utility is dead code, but the compile output is already correct for the serial case. This is a completeness gap in the compilation pipeline, not a runtime defect. Severity should be MEDIUM (unused code, feature incomplete).

---

### Finding FG-2 [CRITICAL claimed] -- Slow path produces wrong interleaving order

**Verdict: REAL**

**Rationale:** The algorithmic bug in the slow path is genuine. When `order` excludes interleaved async passes (e.g. only contains the graphics queue) and those async passes produce resources consumed by later-in-order non-async passes, appending all deferred async passes at the end destroys the dependency-respecting interleaving. The example in the finding is correct: an async pass at index 1 writing a resource read by a graphics pass at index 2 would be reordered to position [1, 3] -- pass 2 reads stale data. **However**, the bug is **currently dormant** -- no production caller reaches the slow path (and nothing calls `serial_fallback` at all per FG-1). The finding correctly identifies a real bug; the impact at this moment is not CRITICAL because the code path is unreachable, but it becomes CRITICAL as soon as any integration code calls this with a reduced order. Severity should be **HIGH** (real bug, currently dormant).

---

### Finding FG-3 [HIGH] -- `_passes` parameter is dead but documented as functional

**Verdict: REAL**

**Rationale:** The parameter carries a leading underscore (compiler warning suppressed), is never read in the function body, yet the docstring claims it is "used for index bounds." This is a documented API contract that is not upheld. A caller passing an empty or mismatched `passes` slice receives no validation. The HIGH severity is appropriate -- the lying docstring will cause maintenance confusion and hides a potential source of downstream index-out-of-bounds crashes.

---

### Finding FG-4 [HIGH] -- Duplicate pass indices in output

**Verdict: REAL**

**Rationale:** The `deferred` vector iterates over `async_passes` (not `async_set`), and `serial.contains(idx)` checks against the pre-modification `serial`. When `async_passes` contains duplicate `(PassIndex, String)` entries (a pass eligible for both compute and copy), both duplicates pass the filter and are appended. The result is duplicate `PassIndex` values in the output vector, which at execution time means double-submission of the same GPU pass. Correctness issue confirmed. Severity HIGH is appropriate.

---

### Finding FG-5 [HIGH] -- No contract enforcement for `order` parameter

**Verdict: REAL**

**Rationale:** The API accepts `&[PassIndex]` but produces wildly different output depending on whether `order` is the full topological order (fast path, correct) or a subset like the graphics queue (slow path, buggy). There is no type-level or runtime enforcement of the documented contract. This is a genuine API footgun that will cause bugs as soon as the function gains a production caller. HIGH is appropriate.

---

### Finding FG-6 [MEDIUM] -- `async_set` HashSet construction copy-pasted in 3 places

**Verdict: REAL**

**Rationale:** The exact same pattern `async_passes.iter().map(|(idx, _)| *idx).collect()` appears at lines 4422, 4504, and 4581. If the `async_passes` type representation changes (e.g. from a tuple to a struct), all three sites must be updated. A combined helper (`fn async_pass_set(...) -> HashSet<PassIndex>` with an `is_async_pass` variant) would eliminate the drift risk. MEDIUM is appropriate.

---

### Finding FG-7 [MEDIUM] -- `is_async_pass()` helper exists but is unused by `serial_fallback`

**Verdict: OVERZEALOUS**

**Rationale:** `is_async_pass()` (line 4454) performs an O(n) linear scan for a single pass check. `serial_fallback` needs O(1) bulk lookups via a `HashSet` because it tests every pass in `order`. These are different operations for different performance regimes. The proper unification is FG-6's proposed `async_pass_set()` batch helper, not grafting a linear scan into a hot-path set build. The finding mistakes a performance-driven design choice for code drift. The observation that the two APIs should be kept consistent is fair, but calling it a defect is overzealous.

---

### Finding FG-8 [MEDIUM] -- Tests do not validate the slow path dependency order

**Verdict: REAL**

**Rationale:** The sole slow-path test (`serial_fallback_merges_async_passes_when_order_excludes_them`, line 11843) uses sequential indices `[4, 5]` for async passes and `[0, 1, 2, 3]` for non-async -- the appending-at-end behavior happens to produce the correct result by accident. The test verifies presence and relative non-async order but does NOT verify the relative positioning of async vs. non-async passes. The critical edge case (async pass with lower index than a non-async pass that depends on its output) is not tested, so the bug in FG-2 is not caught. MEDIUM is appropriate -- the test gap is real.

---

## Coverage Gaps Assessment

| Gap | Junior Severity | Verdict |
|-----|-----------------|---------|
| Slow path with interleaved pass indices | HIGH | REAL -- same root cause as FG-2. Gating factor for resolving FG-2. |
| Duplicate async_passes entries | HIGH | REAL -- maps to FG-4. |
| Empty `order` | MEDIUM | REAL -- missed boundary case. |
| Empty `passes` | MEDIUM | OVERZEALOUS -- `_passes` is unused (FG-3). Adding this test before deciding whether to keep or remove the parameter is premature. |
| Order containing ONLY async passes | MEDIUM | REAL -- an untested edge case that would exercise both paths. |
| `order` being `build_async_plan().graphics_queue` | HIGH | REAL -- the exact misuse scenario from FG-2/FG-5. Should be tested with intentionally interleaved indices. |
| Single-element `order` | LOW | REAL -- trivial boundary, would be covered naturally by property-based testing. |
| `usize::MAX` as PassIndex | LOW | OVERZEALOUS -- testing boundary values of `PassIndex.0` is testing the sort implementation, not `serial_fallback` logic. |
| Async passes with inter-dependencies | MEDIUM | REAL -- two async passes where A depends on B being sorted only by `PassIndex.0` is fragile. |
| `order` containing some but not all async passes | MEDIUM | REAL -- the mixed-path scenario is untested and exercises the filter logic. |

---

## Final Summary

| Junior Severity | Junior Count | Upheld | Downgraded | Overturned |
|-----------------|-------------|--------|------------|------------|
| CRITICAL        | 2           | 0      | 1 (HIGH: FG-2) | 1 (FG-1) |
| HIGH            | 3           | 3      | --         | --         |
| MEDIUM          | 3           | 2      | --         | 1 (FG-7)   |
| LOW             | 1           | 1      | --         | --         |

**Bottom line:** 6 of 8 findings are genuine. The two critical findings are both over-reached: FG-1 misdiagnoses structure as impact (the compile output is already correct for serial dispatch -- the function just isnt wired in), and FG-2 identifies a real but dormant bug. The actionable issues are FG-2 (fix the slow path merge algorithm), FG-3 (remove or validate `_passes`), FG-4 (deduplicate through `async_set`), FG-5 (strengthen the API contract), FG-6 (extract the helper), and FG-8 (add proper interleaved-index tests). FG-1 and FG-7 should be dismissed.
