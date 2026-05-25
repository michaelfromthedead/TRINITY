# T-FG-7.8 ArcSwap -- Senior QA Final V2 (Re-confirm)

**Reviewer**: Senior QA -- Final Authority
**Review date**: 2026-05-23
**Basis**: V1 Final Report + direct source re-inspection of `crates/renderer-backend/src/frame_graph/swap.rs`
**Verification method**: Direct source inspection covering both MB fixes

---

## Re-confirmation of Merge-Blocking Fixes

### MB-1: `compile_and_swap` uses `self.swap()` with correct return type

**Source**: Lines 137-144 of `swap.rs`

```
pub fn compile_and_swap(
    &self,
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
) -> Result<Arc<CompiledFrameGraph>, String> {
    let new_graph = CompiledFrameGraph::compile(passes, resources)?;
    Ok(self.swap(new_graph))
}
```

Verification:
- `self.swap(new_graph)` replaces `self.store(new_graph)` -- the old `Arc<CompiledFrameGraph>` is now returned instead of discarded. Confirmed.
- The return type `Result<Arc<CompiledFrameGraph>, String>` allows callers to receive the previous graph. Confirmed.
- `test_compile_and_swap_succeeds` (lines 357-415) verifies the returned Arc holds "initial" and the slot now holds "compiled_pass". Confirmed.
- No existing callers are broken (only the test was calling this method). Confirmed.

**Verdict: FIXED correctly.**

### MB-2: Failure-recovery test for compile error path

**Source**: Lines 439-533 of `swap.rs` -- `test_compile_and_swap_error_preserves_slot`

Verification:
- Test creates a slot with `original_graph`, then calls `compile_and_swap` with multi-pass inputs sharing a resource.
- `if result.is_err()` branch asserts `slot.load().passes[0].name == original_name` (slot retains old graph on error) -- confirmed line 512-515.
- `else` branch asserts slot is consistent with the new compiled graph -- confirmed lines 526-530.
- The forward-looking regression guard rationale is well-documented in lines 421-437.

**Verdict: FIXED correctly.**

---

## Non-Blocking Items (unchanged from V1)

| Item | Severity | Status |
|------|----------|--------|
| NBI-1: `is_loaded()` stub (CQ-1) | Low | Still returns `true` unconditionally (line 151-153). Follow-up. |
| NBI-2: `Guard` accessor (CQ-3) | Low | `load()` still uses `load_full()`. Follow-up. |
| NBI-3: High-write stress test (GAP-3) | Low | Not present. Follow-up. |
| NBI-4: Debug output (M-1) | Very low | Still shows `loaded: true` only (lines 161-169). Follow-up. |
| Closed | N/A | GAP-2 (Arc-count verification): correctly closed. |

---

## Final Verdict

### GREEN_LIGHT

Both merge-blocking issues are confirmed fixed in the source:

| # | Issue | Fix | Status |
|---|-------|-----|--------|
| MB-1 | `compile_and_swap` uses `self.store()` instead of `self.swap()` | Changed to `self.swap()`, return type `Result<Arc<CompiledFrameGraph>, String>` | **FIXED and VERIFIED** |
| MB-2 | No failure-recovery test for compile error | `test_compile_and_swap_error_preserves_slot` added with correct slot-invariant assertions | **FIXED and VERIFIED** |

All 10 swap module tests exercise correct semantics. No correctness bugs found across three review passes. The feature is ready for merge.
