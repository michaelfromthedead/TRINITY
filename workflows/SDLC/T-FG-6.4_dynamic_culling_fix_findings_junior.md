# T-FG-6.4 Dynamic Culling FIX Re-Review -- JUNIOR_QA Findings

**Reviewer:** JUNIOR_QA (FIX re-review)
**Files Reviewed:**
- `crates/renderer-backend/src/frame_graph/mod.rs` (DEV fix cycle changes)

**Context:** SENIOR_QA_FINAL identified 4 REQUIRED items (C-01, C-03/H-01, H-02, H-03) plus 3 RECOMMENDED items (L-01, L-02, L-03). The DEV fix cycle ran. This review verifies each item against the current state of the file.

**Base commit:** `d62c36cc` (before fix)
**Fix commit:** `17d81c90` (after fix)
**Diff size:** 1671 lines added/modified

---

## REQUIRED ITEMS (Must Fix Before Merge)

### R1. Implement `compute_live_output_set` (C-01)

**Verdict: PASS**

The function at line 12474 was a dead stub returning `Vec::new()` unconditionally. The DEV fix implemented the full function:

- Filters resources for `is_history == true`, `ResourceLifetime::History(_)`, or `ResourceLifetime::Imported` (lines 12479-12487).
- When `debug_enabled`, includes resources written by debug passes (passes with non-zero `feature_flags`) at lines 12489-12496.
- Returns the properly computed `Vec<ResourceHandle>`.

**Call site verification:**
- Line 3345: `compute_live_output_set(&passes, &resources, config.debug_outputs_enabled)` -- now passes `config.debug_outputs_enabled` from the `CompilerConfig` parameter instead of the previously hardcoded `false`.

**Test coverage:**
- Added `compute_live_output_set_includes_history_and_imported_resources()` at line 11728 -- verifies history and imported resources appear in the live set.
- Added `compute_live_output_set_includes_debug_pass_writes_when_debug_enabled()` at line 11777 -- verifies debug pass writes are included when `debug_enabled` is `true`.

**No issues found.** The function correctly handles all three categories (history, imported, debug). The return type and signature match the call site. The `debug_enabled` parameter is no longer dead.

---

### R2. Add `compile_with_config()` accepting `CompilerConfig` (C-03 / H-01)

**Verdict: PASS**

The original `compile()` method (old line 3190) had no way to accept a `CompilerConfig`. The DEV fix:

**New `compile_with_config()` method** at line 3306:
- Signature: `pub fn compile_with_config(passes: Vec<IrPass>, resources: Vec<IrResource>, config: &CompilerConfig) -> Result<Self, String>`
- Passes `config.debug_outputs_enabled` to `compute_live_output_set` (line 3345).
- Implements all phases 2-6 including Phase 4.1 (BarrierOptimizer with identity removal + A->B->A cancellation at lines 3327-3330).
- Returns `Ok(CompiledFrameGraph { ... })` with all fields populated (lines 3356-3373).

**Backward compatibility:**
- Original `compile()` at line 3285 preserved and now delegates: `Self::compile_with_config(passes, resources, &CompilerConfig::default())`

**New `apply_runtime_culling()` method** at line 3376:
- Filters `self.order`, `self.scheduled_passes`, and `self.async_timeline` based on `self.runtime_features` and `config.debug_outputs_enabled`.
- Tracks dynamically skipped passes via `CullStats::dynamically_skipped`.

**CompilerConfig doc fix:**
- `debug_outputs_enabled` doc at lines 2976-2980 now correctly references `compute_live_output_set` and `apply_runtime_culling` instead of only "Phase 6".

**No issues found.** The API is clean with backward-compatible `compile()`. The `CompilerConfig` plumbing through to `compute_live_output_set` is correct.

---

### R3. Remove orphan comment (H-02)

**Verdict: PASS**

The stale orphan comment was:
```rust
/// Stub: eliminates dead passes. Currently a no-op.
```

This was deleted in the DEV fix cycle. The diff confirms removal at line 278:
```
-/// Stub: eliminates dead passes. Currently a no-op.
```

Additionally, the other orphan comment was also cleaned up:
```
-// Missing function stubs (recovered from merge, need implementation)
```

The current code at the `compile` / `compile_with_config` area (lines 3285-3306) has clean, accurate documentation.

**Note:** The legitimate "orphan" references at lines 5341-5349 (resources without lifetime entries in `greedy_color_resources`) are NOT orphan comments -- they are functional documentation for the orphan resource coloring logic and should be preserved. These are a different concept from the stale stub comment that was deleted.

---

### R4. Add doc warning to `compile()` about required post-step (H-03)

**Verdict: PARTIAL -- Doc added, assertion not added**

**The REQUIRED doc warning was added:**
- `compile_with_config()` doc at lines 3303-3305 now states:
  "After compilation, call [`apply_runtime_culling`](Self::apply_runtime_culling) with the same config to apply frame-level feature-based culling. Passes are NOT filtered until that method is called."
- `compile()` doc at lines 3292-3305 similarly references the culling step via the link to `compile_with_config`.

**The OPTIONAL recommendation was not implemented:**
The Final Report's "consider" item at section 5 recommendation 4 suggested:
> "Consider adding `#[must_use]` to `apply_runtime_culling` return, or a `debug_assert!` in a future execution method."

Neither `#[must_use]` nor a `debug_assert!` / `BridgeValidator::validate` call was added inside `compile_with_config` or `compile`. The `BridgeValidator::validate` IS called in `to_json_value` (line 3521) but that is the serialization path, not a post-compile assertion in the compilation path.

**Assessment:** The REQUIRED doc warning is complete. The optional assertion is missing but was not marked as REQUIRED in the Final Report. This is acceptable for the FIX cycle but should be tracked as a follow-up.

---

## RECOMMENDED ITEMS (Follow-up -- Not Required for Merge)

### S1. Add `ALL_DEBUG` unit test (L-01)

**Verdict: NOT DONE**

No test was added for `FeatureSet::ALL_DEBUG`. The constant at line 1283 remains untested.

---

### S2. Add blackbox integration test for full pipeline (L-02)

**Verdict: NOT DONE**

`blackbox_dynamic_culling.rs` still only tests `is_pass_live()` directly. No test calls `compile()` or `compile_with_config()` followed by `apply_runtime_culling()` and asserts on `order.len()` or `dynamically_skipped`. The `CompilerConfig` type IS imported and available in the test file's imports, so this is straightforward to add.

---

### S3. Extract shared test harness (L-03)

**Verdict: NOT DONE**

The helper duplication between `#[cfg(test)]` module (`ca()`, `make_tex()`) and blackbox tests (`make_texture()`, `make_buffer()`, etc.) persists. No shared harness was extracted.

---

## Summary

| Item | Priority | Verdict | Detail |
|------|----------|---------|--------|
| R1. `compute_live_output_set` | REQUIRED | **PASS** | Fully implemented with tests |
| R2. `compile_with_config()` | REQUIRED | **PASS** | New method with backward-compat `compile()` |
| R3. Delete orphan comment | REQUIRED | **PASS** | Stale stub comment removed |
| R4. Post-compile assertion/warning | REQUIRED | **PASS (doc)** | Doc warning added; optional assertion pending |
| S1. `ALL_DEBUG` test | RECOMMENDED | **NOT DONE** | -- |
| S2. Blackbox pipeline test | RECOMMENDED | **NOT DONE** | -- |
| S3. Shared test harness | RECOMMENDED | **NOT DONE** | -- |

**Overall FIX re-review verdict: PASS -- 3/4 REQUIRED items fully implemented, 1/4 partially implemented (doc complete, optional assertion deferred). RECOMMENDED items deferred.**

The DEV fix cycle successfully addressed all three blocking REQUIRED items (C-01 implementation, C-03/H-01 `compile_with_config`, H-02 orphan comment deletion). The fourth REQUIRED item's doc warning (H-03) was added; the optional `#[must_use]` or `debug_assert!` was omitted but was not a merge blocker per the Final Report. The three RECOMMENDED items remain unaddressed for follow-up.
