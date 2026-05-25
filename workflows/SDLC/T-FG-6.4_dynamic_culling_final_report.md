# T-FG-6.4 Dynamic Culling -- SENIOR_QA_FINAL Report

**Reviewer:** SENIOR_QA_FINAL
**Review Date:** 2026-05-23
**Verdict:** FIX

---

## 1. Executive Summary

The SANITY report adjudicated 12 findings from the junior review. This final review independently verified 8 REAL findings (1 CRITICAL, 3 HIGH, 3 LOW) and 4 OVERZEALOUS findings. The Dynamic Culling feature itself (`apply_runtime_culling`, `FeatureSet`, `is_pass_live`) is correctly implemented and well-tested. The issues are concentrated in three areas: (1) an incomplete compile-time stub that undermines Phase 6's always-live resource protection, (2) a `CompilerConfig` design disconnect that cannot influence compilation, and (3) doc/comment rot. All four overzealous findings stem from the junior failing to recognize maintained invariants or intentional defensive design.

---

## 2. Final Adjudication

### REAL Findings (Require Action)

| ID | Severity | Subject | Verdict | SANITY Alignment |
|----|----------|---------|---------|------------------|
| C-01 | **CRITICAL** | `compute_live_output_set` stub disables Phase 6 always-live resource protection | **CONFIRMED** | Aligned. Mitigating: Graphics/RT passes survive via belt-and-suspenders check at lines 3131-3138. The `all_unread` check at line 3119 catches the common case. Gap is narrow: compute/copy passes writing to an always-live resource (swap chain, history) that no downstream pass reads. Correctness gap, not crash bug. |
| C-03 | **HIGH** | `compile()` does not accept `CompilerConfig`; `async_schedule` unconditional | **CONFIRMED (downgraded)** | Aligned on downgrade. `async_schedule` produces passive metadata only -- no correctness/crash risk, only wasted CPU. Severity CORRECTED from CRITICAL to HIGH. |
| H-01 | **HIGH** | `debug_outputs_enabled` doc claims Phase 6 protection but config never reaches `compile()` | **CONFIRMED** | Aligned. Point 1 of H-01 is a genuine documentation-vs-behavior mismatch. |
| H-02 | **HIGH** | Stale orphan comment "Stub: eliminates dead passes. Currently a no-op." at line 3189 | **CONFIRMED** | Aligned. Comment at line 3189 is factually wrong (line 3070 performs real work), misplaced (detached from `compile()` at line 3192), and has wrong indentation (missing leading spaces). |
| H-03 | **HIGH** | Unenforced post-compile `apply_runtime_culling()` invariant | **CONFIRMED** | Aligned. `runtime_features` is passive (set at line 2979). Only consulted by `apply_runtime_culling()` at line 3277. No assertion, no builder chaining, no doc warning on `compile()`. A caller who sets `graph.runtime_features = ...` but omits the culling call gets no notification that ALL debug passes will execute. |
| L-01 | **LOW** | `ALL_DEBUG` constant has zero test coverage | **CONFIRMED** | Aligned. `ALL_DEBUG` at line 1283 equals `DEBUG_WIREFRAME | DEBUG_OVERLAY | DEBUG_PROFILER` but no test verifies this equality. If bits are reassigned, the constant silently diverges. |
| L-02 | **LOW** | No blackbox test exercises `compile()` + `apply_runtime_culling()` pipeline | **CONFIRMED** | Aligned. All blackbox tests in `blackbox_dynamic_culling.rs` use the `is_pass_live()` predicate directly. The unit tests at lines 11395-11583 DO cover the full pipeline. Gap is at the blackbox boundary only. |
| L-03 | **LOW** | Test helper duplication between `#[cfg(test)]` module and blackbox tests | **CONFIRMED** | Aligned. `ca()`/`make_tex()` in `#[cfg(test)]` (lines 11367-11393) vs. `make_texture()`/`make_buffer()`/`graphics_pass()`/`compute_pass()` in blackbox (lines 56-136). Near-identical interfaces, two change points. |

### OVERZEALOUS Findings (No Action Required)

| ID | Severity | Subject | Verdict | Rationale |
|----|----------|---------|---------|-----------|
| C-02 | CRITICAL | `apply_runtime_culling()` orphans `eliminated_passes` | **DISMISSED** | Doc at line 2959 explicitly scopes `eliminated_passes` to Phase 6. Dynamic culling tracked separately via `CullStats::dynamically_skipped` (line 2729). `order` is correctly pruned at line 3295. No invariant violated. |
| M-01 | MEDIUM | `n_skipped` underflow when PassIndex in order but not passes | **DISMISSED** | Invariant maintained by `compile()` construction path (lines 3224-3225, 3238). Every `order` entry corresponds to a `passes` entry by construction. Purely theoretical concern. |
| M-02 | MEDIUM | `FeatureSet::Display` only handles first 4 bits | **DISMISSED** | Hex fallback at lines 1341-1345 for unknown bits is standard defensive design. Strictly better than panicking or silently dropping bits. Not a maintenance trap. |
| L-04 | LOW | `FeatureSet::is_empty()` is unused | **DISMISSED** | `is_empty()` IS called in the `Display` impl at line 1333. The claim is factually incorrect. |

---

## 3. Source Code Verification

### 3.1 C-01: `compute_live_output_set` -- Confirmed Stub

Location: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`

- Line 12142: `fn compute_live_output_set(...)` returns `Vec::new()` unconditionally.
- Line 12135: Surrounded by comment `// Missing function stubs (recovered from merge, need implementation)`.
- Line 3225: `compile()` calls it with `false` hardcoded for `_debug_enabled`.
- Line 3112: `writes_in_live` guard never triggers because `live_outputs` is always empty.
- Line 3223-3227: Phase 6 comment claims "always-live output set (swap chain, history, debug resources)" but none of these are actually computed.
- Lines 3129-3136: Belt-and-suspenders resurrects Graphics/RT passes unconditionally, mitigating the most common case.
- Lines 3117-3124: `all_unread` check catches the ordinary case (pass writing resource no downstream reads).

### 3.2 C-03: `compile()` Signature -- Confirmed

Location: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`

- Line 3192: `pub fn compile(passes: Vec<IrPass>, resources: Vec<IrResource>) -> Result<Self, String>` -- no `CompilerConfig` parameter.
- Line 3213: `async_schedule(&order, &passes, &edges)` runs unconditionally, producing passive metadata only.
- Line 1398: `Async compute available` on `CompilerConfig` only used by the post-compile `log_async_compute_warnings` advisory.

### 3.3 H-02: Orphan Comment -- Confirmed

Location: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`

- Line 3189: `/// Stub: eliminates dead passes. Currently a no-op.` -- factually wrong (line 3070 `eliminate_dead_passes` performs real Phase 6 work), misplaced (floats between `compile()`'s doc comment and the function), and has inconsistent indentation (missing leading spaces vs. surrounding code at line 3184-3187, 3192).

### 3.4 H-03: Unenforced Invariant -- Confirmed

Location: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`

- Line 2979: `pub runtime_features: FeatureSet` defaults to `FeatureSet::NONE`.
- Line 3277: `apply_runtime_culling()` reads `self.runtime_features` and `config.debug_outputs_enabled` to build the feature set.
- No assertion, no doc warning, no builder chaining enforces the required post-compile call.

### 3.5 L-02: Blackbox Integration Gap -- Confirmed

Location: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/tests/blackbox_dynamic_culling.rs`

- All 12 test sections use `is_pass_live(pass, features)` predicate directly (lines 506-507, 530-531, 558-559, etc.).
- No test calls `graph.apply_runtime_culling(&config)` and asserts on `graph.order.len()`.
- The full pipeline IS exercised at the unit level (lines 11395-11583 in `mod.rs`).
- Blackbox gap: no external-crate test verifies the structural mutation of `order`/`scheduled_passes`/`async_timeline`.

---

## 4. Risk Assessment

### Currently Deployed Risk

The Dynamic Culling feature as it stands will function correctly in production for the intended use case: a caller who calls `graph.apply_runtime_culling(&config)` after `compile()` will see correct culling behavior. The order is pruned correctly, scheduled passes are filtered correctly, and the `dynamically_skipped` count accurately reflects the number of culled passes.

The deployed risk is therefore **LOW** for the common path:
- A caller who reads the `CompiledFrameGraph` doc and calls `apply_runtime_culling` will get correct behavior.
- Phase 6 always-live resource protection is degraded (the stub) but the belt-and-suspenders check protects Graphics/RT passes.

The risk is **MEDIUM** for the uncommon path:
- A caller who sets `graph.runtime_features` but omits `apply_runtime_culling()` will silently execute ALL debug passes -- no error, no warning. This is the most dangerous footgun.
- A compute/copy pass writing to a swap chain or history resource that no other pass reads could be incorrectly eliminated by Phase 6.

### Regression Risk of Fixes

| Fix | Regression Risk | Mitigation |
|-----|----------------|------------|
| Implement `compute_live_output_set` | LOW. Adding to `live_outputs` can only KEEP passes alive (more conservative). Cannot cause a pass to be incorrectly eliminated. | Add unit test that history resources are in the output set. |
| Accept `CompilerConfig` in `compile()` | LOW-MEDIUM. Changing a public function signature is a breaking API change. | Add a new `compile_with_config()` method; keep existing `compile()` with default config for backward compatibility. |
| Fix orphan comment | NONE. Cosmetic only. | Trivial removal. |
| Add assertion for culling call | LOW. Panic post-compile if culling not applied. Same pattern as existing assert. | Assert at the start of the first execution pathway (not in common construction). |

---

## 5. Recommendations

### REQUIRED (Before Merge)

1. **Implement `compute_live_output_set` (C-01)**
   - Walk resources and include any with `r.is_history == true` or matching swap chain semantics.
   - Wire `debug_enabled` from `CompilerConfig::debug_outputs_enabled` through a new `compile_with_config()` entry point.
   - Add a `#[test]` verifying that history resources and swap-chain surfaces appear in the returned set.
   - File: `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs`, functions at lines 12142 and 3192.

2. **Add `compile_with_config()` accepting `CompilerConfig` (C-03 / H-01)**
   - New method signature: `pub fn compile_with_config(passes, resources, config: &CompilerConfig)`.
   - Pass `config.debug_outputs_enabled` to `compute_live_output_set`.
   - Keep existing `compile()` calling `compile_with_config()` with `CompilerConfig::default()` for backward compatibility.
   - Correct the doc on `CompilerConfig::debug_outputs_enabled` (line 2887-2888) to reflect that it affects `compute_live_output_set` in Phase 6 AND `apply_runtime_culling`.

3. **Remove orphan comment (H-02)**
   - Delete the detached line 3189: `/// Stub: eliminates dead passes. Currently a no-op.`

4. **Add doc warning to `compile()` about required post-step (H-03)**
   - Add to the `compile()` doc comment: "After compilation, call `apply_runtime_culling(&config)` to apply frame-level feature culling. Passes are NOT filtered until this method is called."
   - Consider adding `#[must_use]` to `apply_runtime_culling` return, or a `debug_assert!` in a future execution method.

### RECOMMENDED (Follow-up)

5. **Add `ALL_DEBUG` unit test (L-01)**
   - One-liner: `assert_eq!(FeatureSet::ALL_DEBUG, FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY | FeatureSet::DEBUG_PROFILER);`
   - Protects against bit reassignment drift.

6. **Add blackbox integration test for full pipeline (L-02)**
   - New test in `blackbox_dynamic_culling.rs`: call `compile()` then `apply_runtime_culling()` then assert on `order.len()`, `scheduled_passes.len()`, and `dynamically_skipped`.
   - Use `CompilerConfig` directly (it is public) rather than the `is_pass_live` predicate.

7. **Extract shared test harness (L-03)**
   - Move `ca()` and `make_tex()` from `#[cfg(test)]` module and the blackbox helpers into a shared test support crate, or re-export from a `pub mod test_support` gated by `#[cfg(test)]`.
   - This creates a single change point for `IrPass` constructor signature changes.

---

## 6. Quality Metrics

| Metric | Value |
|--------|-------|
| REAL findings | 8 (67%) |
| OVERZEALOUS findings | 4 (33%) |
| Signal quality | Solid. Junior correctly identified genuine issues but consistently overstated severity. |
| SANITY adjudication accuracy | 12/12 findings correctly classified. All four overzealous designations correctly identify maintained invariants or intentional design. |
| Code quality | Good. Dynamic culling feature works correctly. Clean architecture with Phase 6 static elimination and frame-level dynamic culling as separate concerns. |
| Test coverage (dynamic culling) | 4 unit tests exercise the full pipeline; 12 blackbox tests exercise the predicate at the public API boundary. |
| Documentation issues | 3 findings: orphan comment, misleading doc claim, missing post-step warning. |

---

## 7. Final Verdict: FIX

The codebase is well-structured and the Dynamic Culling feature is functionally correct for the intended production path. The issues are concentrated in completeness (the `compute_live_output_set` stub), API design (`CompilerConfig` not reaching `compile()`), and documentation accuracy (orphan comment, misleading doc, missing warning).

**4 REQUIRED fixes** (C-01 implementation, C-03/H-01 `compile_with_config`, H-02 comment removal, H-03 doc warning) must be completed before merge. These are all low-regression-risk changes: the implementation fix is strictly additive (more passes survive Phase 6, which is conservative), the API addition is backward-compatible via a new method, and the doc fixes are cosmetic.

**3 RECOMMENDED fixes** (L-01 test, L-02 blackbox integration, L-03 test harness) should be scheduled as follow-up work but do not block the merge.

**Signal quality:** The SANITY review correctly identified 8/12 real issues and 4/4 overzealous ones. The junior's signal was solid but consistently overstated severity -- a pattern to note for calibration but not a concern. This review is complete and the code is ready for `FIX` with the four required items above.
