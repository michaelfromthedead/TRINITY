# T-FG-4.4 BarrierOptimizer FIX Re-review -- JUNIOR_QA Findings

**Reviewer**: JUNIOR_QA  
**Target**: `crates/renderer-backend/src/frame_graph/mod.rs`  
**Commit**: `17d81c90` (HEAD)  
**Tests Passing (inline)**: 383  
**Scope**: BarrierOptimizer wiring into `compile_with_config`, CullStats refactor, runtime culling, serial_fallback, resource allocator H-04 fix.

---

## SUMMARY

The DEV correctly wired BarrierOptimizer into `compile_with_config` (Phase 4.1 pipeline), implemented the core identity-removal and A-B-A cancellation logic, and added `compute_live_output_set` (no longer a stub). 383 inline tests pass. However, the blackbox contract test suite was **not compiled or run** -- 6 blackbox test files fail to compile, and the BarrierOptimizer public API has both structural and semantic gaps relative to its blackbox contract.

**Grade: BETA** -- The inline pipeline integration is solid, but the public API surface is incomplete and the blackbox test suite is red.

---

## ISSUES FOUND

### CRITICAL: BarrierOptimizer public API does not match blackbox contract (30+ compilation errors)

The blackbox test file `crates/renderer-backend/tests/blackbox_barrier_opt.rs` (1,243 lines, 45 tests) was authored against a specification that includes **three** elimination rules. The DEV implementation only provides **two** of the expected passes, and uses a different calling convention.

#### 1a. Instance method vs. associated function -- all 45 tests broken

The blackbox tests call `opt.optimize(input)` as an instance method:

```rust
// blackbox_barrier_opt.rs:51
let opt = BarrierOptimizer::new();
let result = opt.optimize(vec![]);
```

But the implementation defines `optimize` as an associated function without `&self`:

```rust
// mod.rs:2802
pub fn optimize(barriers: &[BarrierTuple]) -> Vec<BarrierTuple> {
```

Rust cannot call an associated function as a method. Every test call `opt.optimize(...)` produces `error[E0599]: no method named 'optimize' found`.

Similarly, `remove_identities` is defined as an associated function but called via `opt.remove_identities(input)` throughout.

**Fix**: Either add `&self` parameter to both methods (making them instance methods), or add a `new()` constructor. Simplest: change to `fn optimize(&self, barriers: &[BarrierTuple]) -> Vec<BarrierTuple>` and `fn remove_identities(&self, barriers: &[BarrierTuple]) -> Vec<BarrierTuple>`.

#### 1b. Missing `BarrierOptimizer::new()` constructor -- 45 sites

The struct derives `#[derive(Default)]` which provides `BarrierOptimizer::default()`, but the blackbox tests universally call `BarrierOptimizer::new()`. 45 occurrences.

**Fix**: Add `pub fn new() -> Self { Self }` (or wrap `Default`).

#### 1c. Type mismatch: 6-tuple vs. 5-tuple

The blackbox tests use a 5-tuple type:

```rust
type Barrier5 = (PassIndex, PassIndex, ResourceHandle, ResourceState, ResourceState);
```

But the implementation uses `BarrierTuple` which is a 6-tuple including `EdgeType`:

```rust
pub type BarrierTuple = (
    PassIndex, PassIndex, ResourceHandle, EdgeType,
    ResourceState, ResourceState,
);
```

Every call site passes `Vec<Barrier5>` but the functions accept `&[BarrierTuple]`. Zero of the 45 blackbox tests would type-check.

**Fix**: Either make the blackbox tests use 6-tuples, or change the public API to accept a 5-tuple and internally default EdgeType. Recommend the former since EdgeType is load-bearing for GPU barrier semantics.

#### 1d. Missing read-read elimination (Rule 2) -- 14 tests fail at runtime

The blackbox contract specifies **three** elimination rules (see contract in `blackbox_barrier_opt.rs:8-12`):

| Rule | Description | Blackbox Tests | Implemented? |
|------|-------------|----------------|--------------|
| 1 | Same-state elimination (before == after) | 4 tests | YES - `remove_identities` |
| 2 | Read-read elimination (both read-only) | 14 tests | NO |
| 3 | Deduplication (same from/to/resource triple) | 11 tests | NO |

The DEV's `cancel_adjacent_pairs` is a **different** optimization (cancels A-B-A adjacent pairs on the same key), which is not in the blackbox contract. The contract expects read-read pairs (`ShaderRead -> ShaderRead`, `VertexBuffer -> ShaderRead`, etc.) to be eliminated, and exact duplicates to be collapsed.

Even if the API mismatch (#1a-1c) were fixed, 25 of the 45 blackbox tests would fail at runtime because read-read elimination and deduplication are not implemented.

**Fix**: Implement Rule 2 (read-read: when both `before` and `after` are in the read-only set) and Rule 3 (dedup: collapse identical `(from, to, resource)` triples, keep first).

#### 1e. A-B-A cancellation is not in the blackbox contract

The `cancel_adjacent_pairs` logic (Pass 2 + Pass 3) is not tested by any blackbox test. It is also logically distinct from dedup: A-B-A cancellation removes two barriers, while dedup collapses one of two identical entries. Both are valid optimizations but they operate on different patterns.

**Recommendation**: Add inline tests for A-B-A cancellation (the `tfg47_barrier_a_to_b_to_a_cancellation` and `tfg47_barrier_cascading_cancellation` tests document this correctly), and consider whether the blackbox contract should be updated to include this rule.

---

### MODERATE: `culled_pass_count` field removed from CullStats struct -- 29 compilation errors

The DEV removed the `culled_pass_count` field from the `CullStats` struct (replacing the Display output with `dynamically_skipped`). The `CullStats` struct has `passes_eliminated` which serves the same purpose, but the blackbox test file `crates/renderer-backend/tests/blackbox_cull_stats.rs` references `compiled.cull_stats.culled_pass_count` at 29 sites.

```rust
// blackbox_cull_stats.rs:58
compiled.cull_stats.culled_pass_count > 0,
//                        ^^^^^^^^^^^^^^^ field does not exist
```

The JSON bridge output DOES preserve `"culled_pass_count"` as an alias for `passes_eliminated` (line 3515), so any JSON-based tests would pass. Only the direct struct field access is broken.

**Affected file**: `crates/renderer-backend/tests/blackbox_cull_stats.rs` (29 occurrences across the file)  
**Fix**: The blackbox tests need to reference `passes_eliminated` instead, or the struct needs a compatibility accessor. Recommend adding a `#[allow(deprecated)]` getter or simply updating the tests.

---

### MODERATE: Non-existent `mock_pass_compute` and `mock_pass_graphics` functions -- 13 compilation errors

The blackbox test `crates/renderer-backend/tests/blackbox_alias_policy.rs` imports `mock_pass_compute` and `mock_pass_graphics` from `renderer_backend::frame_graph`:

```rust
use renderer_backend::frame_graph::{
    ..., mock_pass_compute, mock_pass_graphics, ...
};
```

These functions do not exist in the crate's public API. The mocks module (`mocks.rs`) provides `MockPassNode` with builder methods (`.graphics()`, `.compute()`, `.copy()`) and only re-exports `mock_resource_buffer` and `mock_resource_texture`.

**Affected file**: `crates/renderer-backend/tests/blackbox_alias_policy.rs`  
**Fix**: Either add `mock_pass_compute` and `mock_pass_graphics` convenience functions to `mocks.rs` and re-export them, or update the test to use `MockPassNode::graphics()` / `MockPassNode::compute()`.

---

### MODERATE: `compilation_time_us` field does not exist on `CompiledFrameGraph` -- 20+ errors

The blackbox test `blackbox_chained_opt.rs` expects `CompiledFrameGraph` to have a public field `compilation_time_us: u64`, including custom pass structs (`StampPass`, `SentinelPass`) that operate on it:

```rust
// blackbox_chained_opt.rs:54
graph.compilation_time_us += self.0;
```

The actual `CompiledFrameGraph` struct does not have this field. The test appears to be written against a future version of the FrameGraphCompiler that supports pass composability.

**Affected file**: `crates/renderer-backend/tests/blackbox_chained_opt.rs`  
**Fix**: Either add `compilation_time_us: u64` to `CompiledFrameGraph` (defaulting to 0), or document that this is a planned feature and exclude the test file from the build.

---

### MODERATE: `.stats` field renamed to `.cull_stats` -- 6 compilation errors

Multiple blackbox tests reference `graph.stats`:

```rust
// blackbox_cull_stats.rs (and others)
graph.stats.passes_total
```

The actual field is `graph.cull_stats`. This is a rename from a previous iteration.

**Affected files**: `blackbox_cull_stats.rs`, `blackbox_regression.rs`  
**Fix**: Update references from `.stats` to `.cull_stats`.

---

### LOW: Missing `FrameGraphCompiler` type -- 3 compilation errors

Blackbox tests import `FrameGraphCompiler` from `renderer_backend::frame_graph`:

```rust
use renderer_backend::frame_graph::FrameGraphCompiler;
```

The crate's public API provides `CompiledFrameGraph::compile()` and `CompiledFrameGraph::compile_with_config()` as associated functions. There is no `FrameGraphCompiler` struct.

**Affected files**: `blackbox_alias_policy.rs`, `blackbox_regression.rs`  
**Fix**: Remove the import and use `CompiledFrameGraph::compile()` directly, or create a `FrameGraphCompiler` builder struct if the API design requires it.

---

### LOW: `enable_barrier_opt` field missing from `CompilerConfig` -- 1 error

```rust
// blackbox_cull_stats.rs
CompilerConfig { enable_barrier_opt: true, .. }
```

**Fix**: Add `enable_barrier_opt: bool` to `CompilerConfig` or remove from the test.

---

### LOW: `ResourceState` -> `ResourceHandle` type error in iterators -- 4 errors

Some blackbox tests use iterator chains that produce `ResourceState` but type-annotate the output as `Vec<ResourceHandle>`:

```rust
error[E0277]: a value of type `Vec<ResourceHandle>` cannot be built
  from an iterator over elements of type `ResourceState`
```

**Affected files**: `blackbox_barrier_opt.rs`, `blackbox_cull_stats.rs`  
**Fix**: Correct the iterator chain or type annotation.

---

## WHAT WORKS CORRECTLY

| Area | Status | Details |
|------|--------|---------|
| Inline tfg47_* tests | PASS (5/5) | BarrierOptimizer identity removal, A-B-A cancellation, cascading, non-redundant preservation, empty input |
| Runtime culling tests | PASS (6/6) | apply_runtime_culling, debug/production distinction, FeatureSet filtering, JSON export, mixed passes counting, live output set computation |
| serial_fallback tests | PASS (8/8) | Empty async, order preservation, merge, mixed passes, dedup, slow-path interleave, producer-consumer chain, stress |
| H-04 orphan aliasing | PASS (4/4) | Empty lifetimes, all-imported, history ring buffer wrap, mixed tex2d/tex3d/buffer |
| Live output set | PASS (2/2) | History+imported included, debug pass writes included when debug_enabled=true |
| compile_with_config pipeline | PASS | BarrierOptimizer correctly integrated as Phase 4.1, config.debug_outputs_enabled wired to compute_live_output_set |
| CullStats Display format | PASS | JSON export includes dynamically_skipped, culled_pass_count alias preserved in JSON |
| greedy_color_resources | FIXED | Now accepts lifetimes param, assigns high colors to orphans (no-GPU-hazard guarantee) |
| TextureCube allocation | FIXED | Previously fell through to `_ => {}` (no-op), now correctly allocated |
| PhysicalTexture PartialEq | FIXED | Delegates to `compatible_with()` which intentionally excludes handle from comparison |
| debug_assertions unreachable | FIXED | Guards added for unexpected greedy_color_resources failure paths |

---

## SUMMARY OF COMPILATION ERRORS BY FILE

| Test File | Error Count | Root Cause |
|-----------|-------------|------------|
| `blackbox_barrier_opt.rs` | ~38 | API mismatch (new/optimize/5-tuple), missing read-read + dedup rules |
| `blackbox_cull_stats.rs` | ~17 | `.culled_pass_count` -> `.passes_eliminated`, `.stats` -> `.cull_stats` |
| `blackbox_alias_policy.rs` | ~13 | Missing `mock_pass_compute`/`mock_pass_graphics` exports, missing `FrameGraphCompiler` |
| `blackbox_chained_opt.rs` | ~20 | Missing `compilation_time_us` field on `CompiledFrameGraph` |
| `blackbox_regression.rs` | ~29 | `.culled_pass_count`, `.stats`, `FrameGraphCompiler` |
| `blackbox_scheduled_pass.rs` | ~1 | `.culled_pass_count`, `BarrierOptimizer::new` |
| `blackbox_async2.rs` | ~4 | `AsyncExecutionPlan` missing `PartialEq`/`Default` |

**Total: ~122 compilation errors in 7 blackbox test files**

Note: All 383 inline tests pass. The blackbox errors are entirely in `tests/*.rs` files, not in `src/`.

---

## RECOMMENDATIONS

### Required before merge (in priority order):

1. **Fix BarrierOptimizer public API surface**: Add `new()` constructor, convert methods to instance methods (`&self`), or make the associated functions callable from blackbox tests. A `fn new() -> Self { Self }` plus `fn optimize(&self, barriers: &[BarrierTuple]) -> Vec<BarrierTuple>` is the simplest fix.

2. **Fix type mismatch**: Either update `blackbox_barrier_opt.rs` to use 6-tuples (including `EdgeType`) or add a 5-tuple convenience wrapper.

3. **Implement read-read elimination (Rule 2)**: Add a pass that checks if both `before` and `after` are in the read-only state set (`VertexBuffer, IndexBuffer, IndirectArgument, DepthStencilReadOnly, ShaderRead, TransferSrc`). Barriers matching this pattern are no-ops for GPU synchronization and should be removed.

4. **Implement deduplication (Rule 3)**: Add a `seen: HashSet<(PassIndex, PassIndex, ResourceHandle)>` set that tracks already-seen triples. Skip any barrier whose triple is already in `seen`. This must run AFTER Rules 1 and 2 so that a same-state entry skipped by Rule 1 does not consume the triple slot.

5. **Fix CullStats references**: Add `passes_eliminated` accessor or update `blackbox_cull_stats.rs` to reference the field by its current name.

### Recommended but non-blocking:

6. **Add mock_pass_compute / mock_pass_graphics** convenience functions to `mocks.rs` for backwards compatibility.

7. **Decide on `compilation_time_us`**: Either add the field or clearly mark `blackbox_chained_opt.rs` as a forward-looking test not yet active.

8. **Wire subagent brief memory keys**: Add namespace assignments to `phase1/researcher/inventory` and `phase1/coder/capability-matrix` for memory-as-bus coordination with subsequent SDLC stages.

---

## VERDICT

**RED** -- The `compile_with_config` pipeline integration is correct and the inline tests pass, but the BarrierOptimizer public API has a structural mismatch with its blackbox contract, and 7 blackbox test files (122 errors) fail to compile. The DEV needs to:

1. Align the BarrierOptimizer calling convention with blackbox expectations (instance methods + new())
2. Implement read-read elimination and deduplication rules
3. Fix the 5-tuple vs. 6-tuple type mismatch
4. Update or add accessors for the renamed CullStats field

The fix is moderate in scope (estimated 2-3 hours for a single coder). The core logic (identity removal, A-B-A cancellation, pipeline integration, live output set, runtime culling) is sound and does not need rework.
