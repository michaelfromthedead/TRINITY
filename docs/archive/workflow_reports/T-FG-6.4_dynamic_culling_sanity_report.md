# T-FG-6.4 Dynamic Culling -- SENIOR_QA_SANITY Report

**Reviewer:** SENIOR_QA_SANITY
**Files Reviewed:**
- `crates/renderer-backend/src/frame_graph/mod.rs` (lines 1244-1361, 2720-2780, 2880-3307, 11291-11507, 11940-11946)
- `crates/renderer-backend/tests/blackbox_dynamic_culling.rs` (all 1004 lines)

**Input:** `T-FG-6.4_dynamic_culling_findings_junior.md` -- 11 findings (3C/3H/1M/4L)

---

## Adjudication

### C-01: `compute_live_output_set` is a dead stub -- Phase 6 elimination is silently crippled

**Verdict: REAL**

`compute_live_output_set` at line 11940 returns `Vec::new()` unconditionally, confirming the stub. The `writes_in_live` guard at line 3114 of `eliminate_dead_passes` therefore never triggers. The `_debug_enabled` parameter is passed `false` at line 3223 with no code path ever passing `true`.

**Nuance:** The junior's "exploit scenario" is overblown. Graphics and RayTracing passes always survive Phase 6 via the belt-and-suspenders check at lines 3131-3138 (they are explicitly resurrected). The `all_unread` check at line 3119 catches the common case (pass whose writes are consumed by a downstream reader). The gap only affects a contrived scenario: a Compute or Copy pass writing to a resource that (a) should be "always live" by convention (swap chain, history), and (b) happens not to be read by any other pass. This is possible but unlikely in practice.

The true impact: the "always-live resource" safety net intended by `compute_live_output_set` (including history resources and swap chain surfaces) is non-functional. This is a correctness gap, not a crash bug.

---

### C-02: `apply_runtime_culling()` orphans `eliminated_passes` -- struct invariant is silently violated

**Verdict: OVERZEALOUS**

The doc on `eliminated_passes` at line 2960 explicitly scopes it: "Pass indices that were eliminated as dead **(Phase 6)**." Dynamic culling is a separate mechanism that runs post-compile, and its count is tracked in the separate field `CullStats::dynamically_skipped` (line 2731).

After `apply_runtime_culling`, `self.order` is correctly pruned (line 3293), so any consumer reading `order` to determine submission order will behave correctly. The junior's concern about "a downstream executor relying on `eliminated_passes`" is hypothetical -- there is no evidence such a consumer exists, and if one did, it would be using the wrong field (it should use `order`, which is already correct).

The comparison to `eliminate_dead_passes` (line 3144) is inapt: Phase 6 returns fresh vectors from a constructor; `apply_runtime_culling` mutates an existing graph in-place. Different patterns, different responsibilities. No invariant is violated.

---

### C-03: `compile()` does not accept `CompilerConfig` -- async scheduling cannot be disabled

**Verdict: REAL**

`CompilerConfig` is not in the `compile()` signature (line 3190). `async_schedule` at line 3211 runs unconditionally. The `async_compute_available` field on `CompilerConfig` is only read by `log_async_compute_warnings` (line 1398), a post-compile advisory function.

**Nuance:** The severity of CRITICAL is not justified. `async_schedule` computes passive metadata (a list of async-eligible passes). It does not alter the execution order or the pass structure. On non-TIMELINE_SEMAPHORE hardware, an executor simply ignores `async_timeline` and submits via `order`. There is no correctness or crash risk; only wasted CPU cycles computing unnecessary metadata.

The real gap is that `debug_outputs_enabled` cannot influence Phase 6 (this is C-01's root cause). That gap, however, is C-01's territory and is correctly rated there. As a standalone finding, C-03 documents a design disconnect (config struct exists but isn't accepted by the compilation entry point) whose direct consequences are all minor-to-nonexistent. A more appropriate severity would be HIGH.

---

### H-01: `debug_outputs_enabled` has dual contradictory semantics across the compiler boundary

**Verdict: REAL** (with partial OVERZEALOUS)

**Point 1 (REAL):** `CompilerConfig::debug_outputs_enabled` at lines 2889-2891 claims "When true, debug/output passes are always considered live and will not be eliminated by dead-pass elimination (Phase 6)." This is false because `CompilerConfig` is never passed to `compile()`. Phase 6 (`eliminate_dead_passes`) cannot consult the field. The doc is misleading.

**Point 2 (OVERZEALOUS):** The claim about "inverted-default mental model" is incorrect. `debug_outputs_enabled = false` means "production mode -- skip all debug passes," causing `apply_runtime_culling` to use `FeatureSet::NONE`. This is the documented, intentional behavior. A caller who sets `runtime_features` but leaves `debug_outputs_enabled = false` is contradicting themselves -- the config field explicitly says "I don't want debug output," so ignoring `runtime_features` is correct. No confusion; the semantics are consistent with the field name.

---

### H-02: Stale orphan comment -- "Stub: eliminates dead passes. Currently a no-op."

**Verdict: REAL**

Lines 3186-3188: the doc comment `/// Stub: eliminates dead passes. Currently a no-op.` is demonstrably wrong. `eliminate_dead_passes()` at line 3072 performs real work: it prunes passes, counts freed resources, estimates GPU time savings, and returns `CullStats`. The comment is also syntactically detached (it docs the empty line below it, not the method). This is a clear documentation defect that undermines reader confidence in the compiler's correctness.

---

### H-03: `runtime_features` default of NONE creates a silent footgun

**Verdict: REAL**

The invariant "callers must call `apply_runtime_culling()` after `compile()`" is unenforced. There is no:
- Doc warning on `compile()` mentioning the required post-step
- Builder pattern or method chaining that forces it
- Internal assertion that checks if culling was applied before execution

A caller who reads the doc on `runtime_features` (line 2980, "Defaults to FeatureSet::NONE, which disables all debug passes") may reasonably believe that simply setting `graph.runtime_features = ...` has an effect. It does not -- the field is passive and only consulted by `apply_runtime_culling()`. If the caller omits the culling call, ALL passes (including debug-tagged ones) remain in `order` and will be submitted. This is a genuine API design issue.

---

### M-01: `n_skipped` count is computed incorrectly when `n_before > live_set.len()` is false

**Verdict: OVERZEALOUS**

The concern is that `live_set` is built from `self.passes` while `n_before` is `self.order.len()`, and if a `PassIndex` appears in `order` but not in `passes`, the subtraction could underflow.

The invariant that every `PassIndex` in `order` corresponds to a pass in `passes` is maintained by `compile()` (lines 3224-3225, 3238). The construction path ensures this. There is no mutation path that could break it. The concern is purely theoretical with no evidence of an actual bug.

---

### M-02: `FeatureSet::Display` only handles the first 4 bits

**Verdict: OVERZEALOUS**

Lines 1341-1344: unknown bits above bit 3 are dumped as `0x...` hex. This is standard defensive design, not a maintenance trap. If a future developer adds a new constant, updating `Display` is trivially discoverable by grep or by reading the adjacent code. The fallback ensures the display never silently drops bits, which is strictly better than panicking or ignoring them. This is a strength, not a weakness.

---

### L-01: `FeatureSet::ALL_DEBUG` is defined but has zero test coverage

**Verdict: REAL**

`ALL_DEBUG` at lines 1283-1287 is defined as `DEBUG_WIREFRAME | DEBUG_OVERLAY | DEBUG_PROFILER` but has no dedicated test confirming it equals the OR of its components. Although currently unused in production code, if a future change re-assigns bits (e.g., adding `DEBUG_GPU_TIMINGS` to `ALL_DEBUG`), the constant could silently diverge. A unit test would prevent this. LOW severity is appropriate.

---

### L-02: No blackbox test exercises the full `apply_runtime_culling()` pipeline

**Verdict: REAL**

The blackbox tests in `blackbox_dynamic_culling.rs` use only the `is_pass_live()` predicate directly. They never call `compile()` followed by `apply_runtime_culling()` followed by assertions on `order.len()`. The integration of `apply_runtime_culling` into the pipeline IS covered by unit tests (lines 11319-11507), so this is a coverage gap at the blackbox level only, not a complete absence of testing. LOW severity is appropriate.

---

### L-03: Test helper duplication between unit and blackbox tests

**Verdict: REAL**

The `#[cfg(test)]` module defines `ca()` and `make_tex()` helpers (lines 11291-11317). The blackbox tests define `make_texture()`, `make_buffer()`, `graphics_pass()`, and `compute_pass()` (lines 56-136). These are near-identical copies with minor interface differences. A change to the `IrPass` constructor (e.g., a new required field) would require updating both sets independently. A shared test harness crate would prevent drift. LOW severity is appropriate.

---

### L-04: `FeatureSet::is_empty()` is unused

**Verdict: OVERZEALOUS**

`is_empty()` at lines 1309-1312 IS used -- it is called in the `Display` impl at line 1333: `if self.is_empty()`. The claim that it is "dead code" is factually incorrect. While it has no dedicated unit test, it is exercised transitively by the blackbox `feature_set_display_none` test (which calls `format!("{}", FeatureSet::NONE)`). The `Display` impl calls `is_empty()` on every invocation.

---

## Summary

| Junior Finding | Severity | Verdict | Rationale Summary |
|---|---|---|---|
| C-01 | CRITICAL | **REAL** | Stub confirmed; `writes_in_live` dead code; exploit scenario overblown (Graphics/RT passes always survive Phase 6) |
| C-02 | CRITICAL | **OVERZEALOUS** | Doc scopes `eliminated_passes` to Phase 6; dynamic culling tracked separately via `dynamically_skipped`; `order` is correctly pruned |
| C-03 | CRITICAL | **REAL** | `CompilerConfig` not accepted by `compile()`; severity overstated (async scheduling is passive metadata, not a correctness issue) |
| H-01 | HIGH | **REAL** | Point 1 real (doc claim about Phase 6 is false); Point 2 overzealous (intentional documented behavior) |
| H-02 | HIGH | **REAL** | Comment is demonstrably wrong and misplaced |
| H-03 | HIGH | **REAL** | Unenforced post-compile call invariant; passive field misleads callers |
| M-01 | MEDIUM | **OVERZEALOUS** | Purely theoretical underflow concern; invariant is guaranteed by `compile()` construction path |
| M-02 | MEDIUM | **OVERZEALOUS** | Hex fallback for unknown bits is standard defensive design, not a maintenance trap |
| L-01 | LOW | **REAL** | Valid future-drift concern; constant unverified against its components |
| L-02 | LOW | **REAL** | Blackbox integration gap for `apply_runtime_culling()`; unit tests cover it |
| L-03 | LOW | **REAL** | Genuine helper duplication between test boundaries |
| L-04 | LOW | **OVERZEALOUS** | `is_empty()` IS used in `Display`; "dead code" claim is incorrect |

| Category | Count |
|---|---|
| REAL | **8** (C-01, C-03, H-01, H-02, H-03, L-01, L-02, L-03) |
| OVERZEALOUS | **4** (C-02, M-01, M-02, L-04) |

**Signal quality:** 8/11 findings (73%) are real. The three critical findings (C-01, C-03, H-01) are factually correct but their severity is consistently overstated -- particularly the exploit scenario in C-01 (which ignores the belt-and-suspenders check protecting Graphics/RT passes) and C-03 (which treats passive metadata computation as a correctness bug). The junior correctly identified genuine issues but regularly overestimated impact. Signal quality is solid for the junior level; the four overzealous findings reflect a pattern of seeing "possible bugs" where the code has intentional design or maintained invariants.
