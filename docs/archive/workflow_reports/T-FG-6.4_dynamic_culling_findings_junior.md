# T-FG-6.4 Dynamic Culling -- JUNIOR_QA Findings

**Reviewer:** JUNIOR_QA
**Files Reviewed:**
- `crates/renderer-backend/src/frame_graph/mod.rs`
  - `FeatureSet` (lines 1244-1348)
  - `is_pass_live()` (lines 1356-1361)
  - `eliminate_dead_passes()` (lines 3072-3178)
  - `CompiledFrameGraph::compile()` (lines 3190-3252)
  - `CompiledFrameGraph::apply_runtime_culling()` (lines 3274-3307)
  - Unit tests: `runtime_culling_*` (lines 11319-11507)
  - `compute_live_output_set()` (lines 11940-11946)
- `crates/renderer-backend/tests/blackbox_dynamic_culling.rs` (all 1004 lines)

**Context:** 376 lib tests + 35 blackbox tests pass. This review is adversarial and hypercritical.

---

## CRITICAL

### C-01: `compute_live_output_set` is a dead stub -- Phase 6 elimination is silently crippled

**File:** `mod.rs`, lines 11940-11946

```rust
fn compute_live_output_set(
    _passes: &[IrPass],
    _resources: &[IrResource],
    _debug_enabled: bool,
) -> Vec<ResourceHandle> {
    Vec::new()     // <-- always returns empty
}
```

All three parameters are prefixed with `_`, confirming the function body is deliberately unreachable. This function is called at line 3223 inside `compile()`:

```rust
let live_outputs = compute_live_output_set(&passes, &resources, false);
```

Because `live_outputs` is always empty, the `writes_in_live` guard in `eliminate_dead_passes` (line 3114) **never triggers**. The elimination logic falls through to the `all_unread` check alone, which only catches passes whose write resources are read by nobody. This means:

- A compute/copy pass that writes a resource destined for the swap chain (an "always live" output by convention) CAN still be eliminated if that resource happens not to appear in another pass's `access_set.reads`.
- The `_debug_enabled` parameter is passed `false` unconditionally. There is no code path anywhere in the codebase that calls this function with `true`. The `CompilerConfig::debug_outputs_enabled` field is supposed to guard this path, but it is never plumbed through to `compute_live_output_set`.
- The `log_async_compute_warnings` function (line 1394) is the only function that reads `CompilerConfig`, and it is called outside `compile()`.

**Exploit scenario:** A compute pass producing a swap-chain-readable surface whose handle is not explicitly listed in any downstream pass's `reads` set gets eliminated even though it should always emit. The graph runs silent with a missing frame output.

**Relevant upchain:** `debug_outputs_enabled` in `CompilerConfig` (line 2891) claims it controls whether "debug/output passes are always considered live" during Phase 6. This claim is false. `CompilerConfig` is never passed to `compile()`.

---

### C-02: `apply_runtime_culling()` orphans `eliminated_passes` -- struct invariant is silently violated

**File:** `mod.rs`, lines 3293-3304

```rust
self.order.retain(|pi| live_set.contains(pi));
self.scheduled_passes.retain(|sp| live_set.contains(&sp.pass_index));
self.async_timeline.retain(|pi| live_set.contains(pi));
self.cull_stats.dynamically_skipped = n_skipped;
```

After `apply_runtime_culling()` returns, `self.order` has been pruned but `self.eliminated_passes` is **unchanged**. The doc comment on `eliminated_passes` (line 2960) states it contains "Pass indices that were eliminated as dead (Phase 6)". After dynamic culling, this field is stale -- the removed passes are not appended.

The consequence: any consumer that reads `eliminated_passes` to decide which passes to skip during submission will miss dynamically-culled passes. If a downstream executor relies on `eliminated_passes` to skip GPU submission, dynamically-culled passes will still be submitted.

**Comparable pattern in Phase 6:** `eliminate_dead_passes` (line 3144) correctly populates the returned `eliminated` vector alongside the pruned order.

---

### C-03: `compile()` does not accept `CompilerConfig` -- async scheduling cannot be disabled

**File:** `mod.rs`, line 3190

```rust
pub fn compile(
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
) -> Result<Self, String> {
```

`CompilerConfig` is nowhere in the signature. Inside, `async_schedule` (line 3211) runs unconditionally. The `CompilerConfig::async_compute_available` field (line 2901) is only read in `log_async_compute_warnings()` (line 1398), which is a free function the caller must opt into.

This means:
- There is no mechanism to suppress async compute scheduling for a graph that will run on non-TIMELINE_SEMAPHORE hardware.
- The `async_compute_available` field exists in the config struct but is dead for its primary purpose.

---

## HIGH

### H-01: `debug_outputs_enabled` has dual contradictory semantics across the compiler boundary

**File:**
- Line 2889-2891 (`CompilerConfig`): "When true, debug/output passes are always considered live and will not be eliminated by dead-pass elimination (Phase 6)."
- Line 3275-3278 (`apply_runtime_culling`): When false, all debug passes are skipped (NONE). When true, `runtime_features` controls per-pass liveness.

Two problems:

1. **Phase 6 never reads `debug_outputs_enabled`.** The claim that it "will not be eliminated by dead-pass elimination" is false because `CompilerConfig` is never passed to `compile()`. The field exists only for the post-compile culling step, which is a completely different computation.

2. **Inverted-default mental model.** The `debug_outputs_enabled = false` default causes `apply_runtime_culling` to treat `runtime_features` as NONE, ignoring whatever the caller set on `self.runtime_features`. A caller who sets `graph.runtime_features = FeatureSet::DEBUG_WIREFRAME` but leaves `debug_outputs_enabled = false` (the default) will be confused when wireframe passes are still culled. The field name suggests "should we output debug info?" but it functions as "should we use runtime_features at all?"

---

### H-02: Stale orphan comment -- "Stub: eliminates dead passes. Currently a no-op."

**File:** `mod.rs`, lines 3186-3188

```rust
    
/// Stub: eliminates dead passes. Currently a no-op.

    
```

This comment sits between `impl CompiledFrameGraph {` (3181) and `pub fn compile(` (3190). It is syntactically detached (it docs the empty line below it). Worse, `eliminate_dead_passes()` at line 3072 demonstrably performs real work -- it prunes passes, counts freed resources, estimates GPU time savings, and returns a non-trivial `CullStats`. The comment is:

- **Wrong:** eliminates dead passes is not a no-op.
- **Misplaced:** orphaned between impl block and method.
- **Misleading:** any engineer reading this will doubt the compiler's correctness.

The actual stub is `compute_live_output_set` (line 11940), not `eliminate_dead_passes`. The comment likely migrated during a rename.

---

### H-03: `runtime_features` default of NONE creates a silent footgun

**File:** `mod.rs`, line 3250

```rust
runtime_features: FeatureSet(0),
```

And line 3274-3307: `apply_runtime_culling` must be called separately after `compile()`.

The invariant that "callers must call `apply_runtime_culling()` after `compile()`" is not enforced. There is no:
- Documentation on `compile()` mentioning the required post-step.
- Internal invariant check or assertion.
- Builder pattern that forces the step.

Any caller who compiles a graph and runs it without calling `apply_runtime_culling()` will execute ALL passes including debug-tagged ones, because `order` still contains them. This is a correctness bug waiting to happen, especially since `runtime_features` is documented (line 2979) as "defaults to NONE, which disables all debug passes" -- note the doc says it DISABLES them, but the field alone does nothing without the explicit culling call.

---

## MEDIUM

### M-01: `n_skipped` count is computed incorrectly when `n_before > live_set.len()` is false

**File:** `mod.rs`, lines 3289-3290

```rust
let n_before = self.order.len();
let n_skipped = n_before - live_set.len();
```

`live_set` was computed from `self.passes` (line 3282-3287), not from `self.order`. If `self.order` contains any `PassIndex` that is NOT in `self.passes` (e.g., due to a bug in an earlier phase), then `n_skipped` could underflow (wrap on debug builds) or be negative. In practice, `live_set.len()` should always be <= `n_before` since all passes in `order` come from `passes`, but this is an implicit invariant that is not checked.

---

### M-02: `CullStats::dynamically_skipped` is not serialised in Display impl

**File:** `mod.rs`, line 2734-2743

```rust
impl fmt::Display for CullStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CullStats(passes_total={}, eliminated={}, resources_freed={}, bytes_saved={}, live={}, culled={}, gpu_time_saved={}ms, dyn_skipped={})",
            ...
        )
    }
}
```

Wait -- `dyn_skipped` IS included. Actually let me re-check... Yes it IS. So this finding is invalid. Removing M-02.

---

### M-02 (replaces): `FeatureSet::Display` only handles the first 4 bits

**File:** `mod.rs`, lines 1331-1347

```rust
impl std::fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ...
        if self.contains(Self::DEBUG_WIREFRAME) { parts.push("WIREFRAME"); }
        if self.contains(Self::DEBUG_OVERLAY) { parts.push("OVERLAY"); }
        if self.contains(Self::DEBUG_PROFILER) { parts.push("PROFILER"); }
        if self.contains(Self::ASYNC_COMPUTE) { parts.push("ASYNC_COMPUTE"); }
        let remaining = self.0 & !0b1111;
        if remaining != 0 {
            parts.push(format!("0x{:x}", remaining));
        }
        ...
    }
}
```

Bits 4+ (reserved) dump as "0x..." hex with no named label. If reserved bits are assigned in the future, updating Display is a maintenance trap. This is a minor concern but worth flagging as a maintainability debt.

---

## LOW

### L-01: `FeatureSet::ALL_DEBUG` is defined but has zero test coverage

**File:** `mod.rs`, lines 1283-1287

```rust
pub const ALL_DEBUG: Self = Self(
    Self::DEBUG_WIREFRAME.0
        | Self::DEBUG_OVERLAY.0
        | Self::DEBUG_PROFILER.0
);
```

This constant is neither tested in the unit test suite nor in the blackbox tests. If a future change re-assigns bits incorrectly, `ALL_DEBUG` could silently diverge from the sum of its components.

---

### L-02: No blackbox test exercises the full `apply_runtime_culling()` pipeline

**File:** `tests/blackbox_dynamic_culling.rs`

Every blackbox test that validates dynamic culling uses `is_pass_live()` directly (a building-block free function). The `apply_runtime_culling()` method is tested only in the unit test module inside `mod.rs`. This means:
- There is no blackbox test that calls `compile()` -> `apply_runtime_culling()` -> asserts on `order.len()`.
- The blackbox tests verify the semantics of `is_pass_live` but not its integration into the culling pipeline.
- A regression where `apply_runtime_culling()` silently stopped walking `scheduled_passes` would be caught by unit tests but NOT by blackbox tests.

---

### L-03: Test helper duplication between unit and blackbox tests

**File:**
- `mod.rs` lines 11291-11317: `ca()` and `make_tex()` helpers
- `blackbox_dynamic_culling.rs` lines 56-136: `make_texture()`, `make_buffer()`, `graphics_pass()`, `compute_pass()` helpers

The unit tests define their own `ca()` and `make_tex()` which are near-identical copies of the blackbox test helpers. If the pass constructor API changes (e.g., new required field on `IrPass`), both sets of helpers must be updated. A shared test harness crate or module would prevent drift.

---

### L-04: `FeatureSet::is_empty()` is unused

**File:** `mod.rs`, lines 1309-1312

```rust
pub const fn is_empty(self) -> bool {
    self.0 == 0
}
```

`is_empty()` is used only in the `Display` impl (line 1333). It is not tested and not called in any production code path. Dead code.

---

## Coverage Leak Analysis -- `blackbox_dynamic_culling.rs`

**Verdict: CLEAN. No coverage leak.**

The blackbox tests import **only** public API items from `renderer_backend::frame_graph::*`:

| Symbol | Visibility | Export path |
|--------|-----------|-------------|
| `FeatureSet` | `pub struct` | `frame_graph` |
| `is_pass_live` | `pub fn` | `frame_graph` |
| `CompiledFrameGraph::compile` | `pub fn` | `frame_graph` |
| `IrPass`, `IrResource`, etc. | `pub struct` | `frame_graph` |

No private fields, no internal functions, no module-internal constants are accessed. The `#[cfg(test)]` module is not imported. The tests reference `renderer_backend::frame_graph::is_pass_live` and `renderer_backend::frame_graph::BufferDesc` through their fully-qualified public paths.

The blackbox test file correctly imports from the public crate interface and uses a fresh `IrPass::graphics`/`IrPass::compute` constructor path (not private construction). Section 5 and 6 tests validate that `compile()` preserves passes correctly, which is a legitimate public-contract test.

**However**, the blackbox tests have a coverage gap: they never test `apply_runtime_culling()` as a pipeline step, only `is_pass_live()` as a predicate. `apply_runtime_culling()` is public surface area, so it is eligible for blackbox testing. A full pipeline test (`compile()` -> `apply_runtime_culling()` -> assert `order`) would strengthen coverage.

---

## Summary

| Severity | Count | Key Concern |
|----------|-------|-------------|
| CRITICAL | 3 | `compute_live_output_set` is stub; `eliminated_passes` orphaned; `CompilerConfig` not accepted by `compile()` |
| HIGH | 3 | `debug_outputs_enabled` dual semantics; stale comment; footgun from `runtime_features` being passive |
| MEDIUM | 1 | Display only covers bits 0-3 by name |
| LOW | 4 | Untested `ALL_DEBUG`, no blackbox pipeline test, helper duplication, `is_empty` dead code |

**Total: 11 findings.**

On the positive side: the blackbox test suite is clean with zero coverage leak -- it correctly tests only through the public API surface. The `is_pass_live` predicate is rigorously tested across individual flags, combined flags, double-gated passes, and edge cases.
