# T-FG-5.5 Serial Fallback -- Junior QA Findings

**File reviewed:** `crates/renderer-backend/src/frame_graph/mod.rs`
**Function:** `serial_fallback()` (lines 4493-4543)
**Tests:** 6 tests (lines 11804-11929)
**Date:** 2026-05-23

---

## Severity Legend

| Label | Meaning |
|-------|---------|
| **CRITICAL** | Will produce wrong execution order, duplicate execution, or silent data corruption at runtime. Must fix before integration. |
| **HIGH** | Defects that cause incorrect behavior under specific conditions, or dead code posing as a complete feature. |
| **MEDIUM** | Violations of documented contract, missing validation, or design flaws that will cause bugs as the codebase evolves. |
| **LOW** | Code quality, test coverage gaps, style, or maintainability issues. |

---

## Finding FG-1 [CRITICAL] -- Zero production callers; function is dead code

**Location:** `serial_fallback()` (line 4493) — entire function body.

`grep -rn "serial_fallback" crates/` returns EXACTLY zero hits outside the function definition itself, its docstring example, and its test block. The function is `pub` — it compiles and tests pass, but nothing in the crate calls it. The integration point that would invoke this fallback at runtime has not been written.

**Impact:** The feature T-FG-5.5 is structurally incomplete. The 6 passing tests validate a function that is never reached. If the production path encounters an async-incompatible device, the frame graph compiler will panic, return an error, or silently produce an async schedule that the executor cannot dispatch -- because no code calls `serial_fallback` to flatten the plan.

**Evidence:**
```
% grep -rn "serial_fallback" crates/ --include="*.rs" | grep -v "test\|mod.rs"
(no results outside mod.rs)
```
The function only appears in its own `mod.rs` -- in the docstring, definition, and `#[cfg(test)]` block.

**Action:** The `compile()` method at line 3194 produces a `CompiledFrameGraph` with `async_timeline` (line 3232) and `async_compute_used` (line 3212), but never checks `runtime_features` to decide whether to flatten. Either a new pass in `compile()` must call `serial_fallback` when `FeatureSet::ASYNC_COMPUTE` is absent, or the executor must call it before dispatching. Without this, the feature is a stub.

---

## Finding FG-2 [CRITICAL] -- Slow path produces wrong interleaving order

**Location:** Lines 4523-4540 (the "slow path").

**Scenario:** `serial_fallback` is called with an `order` that excludes some async passes (e.g., `graphics_queue` from `build_async_plan`).

**The bug:** The slow path filters async passes OUT of `order`, collects the missing async passes, sorts them numerically by `PassIndex.0`, then APPENDS ALL of them at the end. This destroys the interleaving that the topological sort established.

**Example:**
- Original topological order: `[0(gfx), 1(async), 2(gfx), 3(async), 4(gfx)]`
- Pass 2 reads a resource that pass 1 writes.
- `graphics_queue` from `build_async_plan`: `[0, 2, 4]`
- `async_passes`: `[(1, "compute"), (3, "compute")]`
- `serial_fallback` slow path returns: `[0, 2, 4, 1, 3]`
- **Execution order:** Pass 2 runs BEFORE pass 1.
- **Result:** Pass 2 reads STALE data -- pass 1 writes the resource at position 3, after it has already been consumed at position 1. Silent data corruption in the rendered frame.

**Root cause:** The function assumes that if a pass is NOT in `order`, it has no ordering constraints relative to passes that ARE in `order`. This assumption is false when `order` is the graphics queue -- async passes were at interleaved positions precisely because of dependency edges.

**Conditions to trigger:** Any of:
- `order` has non-trivial size (>= 2) AND at least one async pass produces a resource consumed by a later-in-topo-order non-async pass.
- The caller passes `build_async_plan(...).graphics_queue` instead of the original topological order.

**Note:** If `order` is the full topological sort (as documented), the fast path at line 4515 returns early and this bug is not triggered. But nothing enforces this contract at the type level -- the bug is a type-incorrect call away.

**Action required:** The slow path must merge deferred passes at their correct topological positions rather than appending. Either:
1. Use the original topological order as the merge template (insert missing async passes at their correct positions).
2. Or, better, change the API to accept `AsyncExecutionPlan` directly and reconstruct by walking the original topo order, inserting passes from graphics and async queues at their natural positions.

---

## Finding FG-3 [HIGH] -- `_passes` parameter is dead but documented as functional

**Location:** Line 4493-4494.

```rust
pub fn serial_fallback(
    _passes: &[IrPass],        // <-- dead parameter
    async_passes: &[(PassIndex, String)],
    order: &[PassIndex],
) -> Vec<PassIndex> {
```

**Docstring (line 4470):**
> * `passes` -- All [`IrPass`]es in the compilation (used for index bounds).

The leading underscore confirms the compiler warning was suppressed -- the parameter is NEVER read. The docstring claims it is "used for index bounds" but no bounds checking, no validation, and no pass index range check exists anywhere in the function body.

**Impact:**
- A caller can pass an empty `passes` slice or passes that have no relationship to the indices in `order`. The function silently produces output with indices that may refer to non-existent passes.
- When a bug does surface (e.g., index out of bounds during execution), the root cause will be far from this function.
- The docstring lies to future readers, creating an API contract that is not upheld.

**Action:** Either:
1. Remove the parameter entirely (breaking change to `pub` API).
2. Add bounds validation: verify every `PassIndex` in the output has `idx.0 < passes.len()`. Return `Result` or panic on mismatch.

---

## Finding FG-4 [HIGH] -- Duplicate pass indices in output

**Location:** Lines 4534-4540.

**Scenario:** `async_passes` contains duplicate `PassIndex` entries (same pass index with different queue types, e.g., a pass that is both compute-eligible AND copy-eligible).

**The bug:**
1. `async_set` (HashSet) deduplicates -- no duplicate detection.
2. The `deferred` vector iterates over `async_passes` (not `async_set`), and the filter `!serial.contains(idx)` checks the current `serial` which has NOT been modified yet.
3. Both duplicate entries pass the filter and are pushed into `deferred`.
4. `serial.extend(deferred)` adds both.

**Result:** The same `PassIndex` appears twice in the output. At execution time, this means the same GPU pass is recorded and submitted twice. For a compute pass with side effects, this could produce double-writes, double-barriers, or GPU-side errors.

**Test coverage:** None of the 6 tests pass duplicate entries in `async_passes`.

**Action:** Filter `async_passes` through the `async_set` when building `deferred`:
```rust
let deferred: Vec<PassIndex> = async_set
    .iter()
    .copied()
    .filter(|idx| !serial.contains(idx))
    .collect();
```

---

## Finding FG-5 [HIGH] -- No contract enforcement for `order` parameter

**Location:** Function signature (line 4496) and docstring (lines 4474-4475).

**Docstring:**
> * `order` -- The topological ordering (from [`topological_sort`]) that already respects all pass dependencies.

**Problem:** The function accepts any `&[PassIndex]` and produces different output depending on whether it is the full topological order or the graphics queue from `build_async_plan`. In the first case the fast path triggers and the result is correct. In the second case the slow path triggers and the result is wrong (Finding FG-2). There is no type-level distinction between the two.

**This is an API footgun.** The function should either:
1. Accept `AsyncExecutionPlan` instead of raw slices, guaranteeing it has the right structure.
2. Accept the original topological order PLUS the async_passes, and always use it as the merge template -- eliminating the slow path entirely.

---

## Finding FG-6 [MEDIUM] -- `async_set` HashSet construction copy-pasted in 3 places

**Locations:**
- Line 4422: `build_async_plan` constructs `HashSet<PassIndex>`
- Line 4504: `serial_fallback` constructs `HashSet<PassIndex>`
- Line 4581: `compute_sync_points` constructs `HashSet<PassIndex>`

All three do:
```rust
let async_set: HashSet<PassIndex> =
    async_passes.iter().map(|(idx, _)| *idx).collect();
```

**Impact:** If the `async_passes` representation changes (e.g., from `Vec<(PassIndex, String)>` to a struct), all three sites must be updated. This is a maintenance trap -- the copy-paste pattern will diverge.

**Action:** Extract a helper:
```rust
fn async_pass_set(async_passes: &[(PassIndex, String)]) -> HashSet<PassIndex> {
    async_passes.iter().map(|(idx, _)| *idx).collect()
}
```
Already there is `is_async_pass()` at line 4454 for single-pass checks -- a batch version should live alongside it.

---

## Finding FG-7 [MEDIUM] -- `is_async_pass()` helper exists but is unused by `serial_fallback`

**Location:** Line 4454 defines `is_async_pass()`. Line 4504 builds a HashSet instead of calling it.

**Impact:** Two code paths for the same check. Inconsistency invites one of them to drift.

---

## Finding FG-8 [MEDIUM] -- Tests do not validate the slow path dependency order

**Test** `serial_fallback_merges_async_passes_when_order_excludes_them` (line 11843) tests the slow path, but:

1. It only verifies that non-async passes maintain their relative ORDER (lines 11869-11878).
2. It does NOT verify where the merged async passes end up RELATIVE to the non-async passes.
3. The async passes `[4, 5]` are numerically higher than all non-async passes `[0, 1, 2, 3]`, so appending at the end happens to produce the correct order by accident (PassIndex is sequential).
4. The critical case -- where an async pass has a LOWER index than the last non-async pass but must run BEFORE a specific non-async pass -- is never tested.

**The existing test passes but does not prove correctness.** It proves only that the function doesn't crash and that non-async passes stay in order. The actual requirement -- "preserving the dependency-respecting positions from the topological order" (docstring line 4466) -- is **not validated** by any test.

---

## Coverage Gaps

| Gap | Severity | Details |
|-----|----------|---------|
| **Slow path with interleaved pass indices** | HIGH (Finding FG-2/FG-8) | Async passes with lower indices than some graphics passes; async passes that produce resources consumed by later graphics passes. Current tests use sequential indices. |
| **Duplicate async_passes entries** | HIGH (Finding FG-4) | Same PassIndex appearing twice in async_passes. Currently produces duplicate output. |
| **Empty `order`** | MEDIUM | `order: []` with non-empty `async_passes`. Would return `deferred` sorted by index -- may be correct, but not tested. |
| **Empty `passes`** | MEDIUM | `_passes: &[]`. Currently no-op (dead param), but if FG-3 is fixed, this should error. |
| **Order containing ONLY async passes** | MEDIUM | Both fast path and slow path behavior for `order == async_passes` is untested. |
| **`order` being `build_async_plan().graphics_queue`** | HIGH (FG-5) | The exact scenario where the slow path runs with real data. Untested. |
| **Single-element `order`** | LOW | Boundary case. |
| **`usize::MAX` as PassIndex** | LOW | Boundary case for `deferred.sort_by_key()`. |
| **Async passes with inter-dependencies** | MEDIUM | Two async passes where A depends on B (B must run before A). Sorting by `PassIndex.0` happens to match if indices are sequentially assigned, but this is a fragile assumption. |
| **Case where `order` already contains some but not all async passes** | MEDIUM | Mixed scenario: some async passes are in the graphics queue, some are not. The filter then deferred logic handles this, but untested. |

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 3 |
| MEDIUM | 3 |
| LOW | 1 |

**Bottom line:** `serial_fallback()` has 6 passing tests that exercise the happy path but miss every edge case that would expose the slow path's broken interleaving logic. The function is also dead code -- nothing calls it -- so even if it were correct, the ASYNC_COMPUTE fallback path is not wired into `compile()`. The `_passes` parameter is a dead giveaway that the integration was sketched but never finished. This T-FG-5.5 should be returned to DEV with these findings before it can be marked complete.
