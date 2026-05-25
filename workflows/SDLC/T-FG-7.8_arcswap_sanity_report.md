# T-FG-7.8 ArcSwap -- Senior QA Sanity Report

**Reviewer**: Senior QA
**Review date**: 2026-05-23
**Re-reviewing**: Junior QA findings from `T-FG-7.8_arcswap_findings_junior.md`
**Source files verified**:
- `crates/renderer-backend/src/frame_graph/swap.rs`
- `crates/renderer-backend/tests/blackbox_frame_graph_swap.rs`

---

## Verdict on Junior QA Findings

The junior review is **thorough and accurate**. No correctness bugs were missed. All three code-quality findings are genuine. Two of three coverage gaps are legitimate; one is overzealous. Details below.

---

## Code Quality Findings -- Sanity Check

### CQ-1. `is_loaded()` is a constant stub -- **REAL** (low severity)

**Assessment**: Verified. Line 152-154 of `swap.rs`:
```rust
pub fn is_loaded(&self) -> bool {
    true
}
```

This is genuinely misleading. The method exists in the public API, has 8 call sites in tests, and always returns `true`. If a future consumer calls `is_loaded()` expecting it to reflect actual state (e.g., a slot that was cleared or never initialized), they will silently get a false positive. The `Debug` implementation also reports `loaded: true` unconditionally (line 168), amplifying the potential confusion.

The junior's two recommendations are sound:
- Option A: Remove the method (simplest, cleanest -- no semantic loss since the slot is always initialized).
- Option B: Implement `Option<ArcSwap<...>>` for genuine lazy initialization (if the feature is planned).

**Recommendation**: Option A (remove) unless there is an explicit roadmap item for lazy slot initialization. If the API is needed for symmetry with a trait, at minimum add a doc comment `#[doc(hidden)]` or prefix with `_` to signal that it is a placeholder.

---

### CQ-2. `compile_and_swap` uses `store` not `swap` -- **REAL** (medium-low severity)

**Assessment**: Verified. Lines 137-145 of `swap.rs`:
```rust
pub fn compile_and_swap(
    &self,
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
) -> Result<(), String> {
    let new_graph = CompiledFrameGraph::compile(passes, resources)?;
    self.store(new_graph);   // <-- calls store(), not swap()
    Ok(())
}
```

The method name promises `swap` semantics (return the old graph) but delivers `store` semantics (silently drop the old graph). This is an API naming violation -- a consumer who reads the method name and sees a public `swap()` method next to it would reasonably expect `compile_and_swap` to return the previous graph.

The junior correctly identifies both remediation paths. The stronger choice is to change the return type to `Result<Arc<CompiledFrameGraph>, String>` and use `self.swap()`, which:
- Makes the API consistent with the standalone `swap()` method
- Allows callers to inspect or keep alive the replaced graph
- Breaks no existing callers (the method is only used in `test_compile_and_swap_succeeds`)

**Recommendation**: Fix before merge. This is the only finding that has a real downstream cost if left unfixed -- renaming it later would be a breaking API change.

---

### CQ-3. `load()` always uses `load_full` -- **REAL** (low severity, was rated one notch too high)

**Assessment**: Verified. Line 100-105 of `swap.rs`:
```rust
pub fn load(&self) -> Arc<CompiledFrameGraph> {
    self.current.load_full()
}
```

The junior's technical analysis is correct: `load_full()` performs an atomic `Arc` refcount increment every call, while `arc_swap::load()` returns a `Guard` that borrows the internal pointer without a refcount bump.

However, the severity rating of "medium-low" slightly overstates the practical impact. In the described use case (once per frame), a single atomic increment is:
- ~20-40ns on modern x86 hardware
- Free of contention since writes (recompilation) happen orders of magnitude less often
- Not compounded in any meaningful way (the returned `Arc` is typically used and dropped within the same function scope)

**Correction**: Downgrade to **low severity**. The observation is valid and the `borrow()` suggestion is a reasonable future optimization, but calling this "medium-low" implies it has a measurable footprint, which it does not for the stated workload. File as a performance follow-up, not a pre-merge concern.

The junior's recommended API expansion (separate `load()` returning `Arc` and `borrow()` returning `Guard`) is a good design. A `Guard`-based default for the hot path would be marginally cheaper, and retaining an `Arc`-returning variant for long-lived handles is useful.

---

## Test Coverage Gaps -- Sanity Check

### GAP-1. No `compile_and_swap` failure-recovery test -- **REAL** (medium severity)

**Assessment**: Verified gap. The sole test (`test_compile_and_swap_succeeds`, line 358) tests only the happy path. The error-propagation semantics on lines 142-144 (slot unchanged on compilation failure) are entirely untested.

This is a legitimate concern because:
- The slot state after a compilation error is an implicit contract: "previous graph is preserved."
- The `?` operator on line 142 short-circuits before `self.store()`, so the slot IS left unchanged, but this is an emergent property of the control flow, not an explicit invariant check.
- If a future refactor accidentally moves `self.store()` before the compile call or wraps it differently, no existing test would catch the regression.

The junior's specific recommendations (duplicate resource handles, pass cycles, mismatched attachments) are good test vectors. Any one of them would suffice to verify the slot-invariant-on-error behavior.

**Recommendation**: Add before merge. This is the most impactful coverage gap because it tests a correctness invariant (slot state preservation) that is currently only implicitly guaranteed by control flow.

---

### GAP-2. No memory-lease / Arc-count verification -- **OVERZEALOUS**

**Assessment**: The junior acknowledges "This is guaranteed by Rust's Arc semantics" and then recommends a test anyway. This is testing a language-level memory safety guarantee, not application logic.

The specific regression scenario described (`Arc::into_raw` without matching `from_raw`) cannot occur in the current codebase because:
- `FrameGraphSlot` never uses `Arc::into_raw` or any raw pointer manipulation.
- The only `Arc` operations are `Arc::new()` (lines 85, 113, 122) and `load_full()` (line 104) -- both standard safe-Rust use.
- There is no `unsafe` block in the entire module (the safety comment on lines 157-160 exists precisely to document why none is needed).

A `Drop` counter or weak-reference test would verify that Rust's standard library `Arc` works as documented -- not that the application code is correct. This is not a meaningful coverage gap for a thin wrapper that delegates entirely to `arc_swap`.

**Recommendation**: Close as "won't fix / not actionable." If a regression test for Arc semantics is desired, file it in the crate-level hardening backlog as a defense-in-depth measure, not as a gap in this feature's coverage.

---

### GAP-3. No stress test under high write frequency -- **REAL** (low severity)

**Assessment**: The junior correctly identifies that the RCU mechanism in `ArcSwap` can accumulate old copies under burst writes. The doc comment (line 16-17) acknowledges this design assumption: "recompilation happens orders of magnitude less often than per-frame reads."

The gap is real because:
- The current stress tests (tests 7a and 7b in the blackbox suite) use modest write counts (10-20).
- A regression test with 1,000 back-to-back stores would catch accidental changes that pin old copies.
- The test is cheap to write and run (bounded memory, fast execution).

However, this is low severity because:
- The doc comment's design assumption is clearly stated and correct for the intended workload.
- `ArcSwap`'s RCU mechanism is well-tested upstream -- old copies are eventually reclaimed when all readers cycle.
- No evidence of memory pressure in the intended use case.

**Recommendation**: Add as a regression guard, but low priority. Good for spotting accidental regressions during refactoring. The specific recommendation (1,000 back-to-back `store()` calls without interleaved reads, verifying bounded memory) is sound.

---

## Issues the Junior Missed

### M-1. `Debug` output is misleadingly constant (very low severity)

The `Debug` implementation (lines 162-171) outputs `FrameGraphSlot { loaded: true }` regardless of the slot's content. This is consistent with `is_loaded()` (which always returns true), but it means `Debug` provides zero diagnostic value -- it never reveals which graph is held, the pass count, or any distinguishing information.

This is tangentially related to CQ-1. If CQ-1 is fixed (either by removing `is_loaded()` or making it meaningful), the `Debug` impl should be updated accordingly. For now it is a minor DX wart, not a correctness issue.

### M-2. No test verifying `swap`-returned `Arc` has correct refcount after chain (very low severity)

The blackbox tests verify that `swap`-returned Arcs remain valid after further mutations (tests 9, 5, 8). They do not explicitly verify that the refcount matches expectations after a chain of N swaps (e.g., that exactly N distinct Arcs exist). This is guaranteed by `Arc` semantics (same argument as GAP-2's overzealous assessment), so this is not a genuine gap -- noted only for completeness.

---

## Summary

| Finding | Verdict | Severity | Action |
|---------|---------|----------|--------|
| CQ-1. `is_loaded()` stub | REAL | Low | Remove or implement properly |
| CQ-2. `compile_and_swap` naming | REAL | Medium-low | **Fix before merge** -- rename or change to `swap` |
| CQ-3. `load()` always uses `load_full` | REAL (downgraded severity) | Low | File as follow-up optimization |
| GAP-1. No failure-recovery test | REAL | Medium | **Add before merge** -- slot invariant on error |
| GAP-2. Arc-count verification | OVERZEALOUS | N/A | Close -- tests language guarantee |
| GAP-3. High-write-frequency stress test | REAL | Low | Add as regression guard, low priority |

### Actions

- **Before merge**: Fix CQ-2 (naming) and add GAP-1 (failure-recovery test).
- **Before merge or follow-up**: Address CQ-1 (remove `is_loaded()` or add real semantics), update `Debug` accordingly.
- **Follow-up**: CQ-3 (`borrow()` method for `Guard`-based access), GAP-3 (stress test).

### Overall

The junior review was well-executed. It correctly identified all actionable issues in the codebase, and the classification was accurate except for the minor over-severity on CQ-3 and the overzealous GAP-2. No correctness bugs were found by either reviewer. The implementation is solid, well-tested, and ready for final sign-off once CQ-2 and GAP-1 are addressed.
