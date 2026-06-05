# T-FG-4.4 BarrierOptimizer -- Senior QA Sanity Report

**Reviewer**: SENIOR_QA_SANITY
**Target**: `crates/renderer-backend/src/frame_graph/mod.rs` lines 2754-2860 (BarrierOptimizer), lines 3190-3252 (compile), lines 2345-2459 (compute_barriers / build_scheduled_passes)
**Tests examined**: `blackbox_barrier_opt.rs`, `whitebox_t_fg_9_5_regression.rs`, `blackbox_regression.rs`, `blackbox_fix7_7.rs`
**Role**: Judicial. Each finding marked REAL or OVERZEALOUS. No new findings.

---

## Verdict by Finding

| # | Severity | Finding | Verdict | Rationale |
|---|----------|---------|---------|-----------|
| 1 | CRITICAL | `optimize()` never called in `compile()` | **OVERZEALOUS (severity)** | REAL that `BarrierOptimizer::optimize()` is absent from `compile()`, but CRITICAL is wrong. The production pipeline independently performs identity elimination (`before != after` at line 2377) and deduplication (`seen` HashSet at line 2377, `HashSet<BarrierTuple>` at line 2410) inside `compute_barriers()` and `build_scheduled_passes()`. Only adjacent-pair cancellation is skipped. Furthermore, `compile()` produces `Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>` (4-tuple), while `BarrierOptimizer` operates on `BarrierTuple` (6-tuple with EdgeType) -- the types are incompatible. Calling `optimize()` would not compile without a conversion layer. |
| 2 | CRITICAL | Blackbox tests use 5-tuple `Barrier5` vs 6-tuple `BarrierTuple` | **REAL** | The blackbox tests define `type Barrier5 = (PassIndex, PassIndex, ResourceHandle, ResourceState, ResourceState)` and pass `Vec<Barrier5>` to `optimize(&[BarrierTuple])`. This is a type error in Rust. The junior's three hypotheses (excluded, wrapper, dynamic construction) miss the actual root cause: **neither the blackbox nor whitebox test files compile against the current production code.** The whitebox file additionally references `compiled.stats.barriers_total`, `compiled.stats.barriers_optimized`, `CompilerConfig::enable_barrier_opt`, and `CompiledFrameGraph::compile_with_config()` -- none of which exist in `mod.rs`. `BarrierOptimizer::new()` does not exist in the source (the struct is a unit struct with no constructor or Default derive). Both test files were written against a spec that was only partially implemented. |
| 3 | HIGH | No deduplication pass in `optimize()` | **OVERZEALOUS (severity)** | REAL that `optimize()` lacks a dedup pass. But the impact is mitigated: `build_scheduled_passes()` (line 2410) uses a `HashSet<BarrierTuple>` which inherently deduplicates exact duplicates. `compute_barriers()` (line 2377) uses a `seen: HashSet<(PassIndex, PassIndex, ResourceHandle)>` for triple-based dedup. The `dedup_exact_duplicate_collapsed` test does document expected behavior that `optimize()` doesn't provide, but that test cannot compile (Finding 2 applies), so the mismatch is academic. Severity should be MEDIUM, not HIGH. |
| 4 | MEDIUM | Non-adjacent cancellation pairs are missed | **REAL** | Correct analysis. `cancel_adjacent_pairs` only checks `(i, i+1)` pairs. Backtrack `i -= 1` handles re-adjacency after removal but does not find interleaved patterns like `A->B, X->Y, B->A`. This is a genuine algorithmic limitation. Severity is appropriate. |
| 5 | MEDIUM | O(n^2) worst-case from `Vec::remove` in tight loop | **REAL** | `Vec::remove` is O(n) due to element shifting. In a degenerate worst case (all pairs cancel), `cancel_adjacent_pairs` degrades to O(n^2). The junior correctly notes this is trivial for expected barrier counts (10s-100s). Could be LOW, but MEDIUM is defensible. |
| 6 | MEDIUM | Missing edge-case test coverage | **REAL (moot)** | Valid observations: shuffled input, empty-after-phase-1, single-element, concurrent, high-cardinality are untested. However, since neither blackbox nor whitebox test file compiles against the current code, coverage gaps are secondary. The entire test suite for BarrierOptimizer is non-functional (see Finding 2). This finding is technically correct but practically moot until the compilation issues are resolved. |
| 7 | MEDIUM | Multiple backup copies of `mod.rs` | **REAL (understated)** | The junior reports 8 copies. Actual count is 16+ versioned files (`mod.rs.work.disabled`, `mod.rs.stable`, `mod.rs.stable2`, `mod.rs.mybackup.disabled`, `mod.rs.manual_fix`, `mod.rs.new`, `mod.rs.final`, `mod.rs.bak.disabled`, `mod.rs.bak.ref`, `mod.rs.bak.tmp`, `mod.rs.corrupted`, `mod.rs.current.disabled`, `mod.rs.edit`, `mod.rs.safe_copy`, `mod.rs.pre_tfg64.disabled`, `mod.rs.backup.disabled`) plus non-versioned artifacts (`temp_edit.rs`, `sedUz56Nx`). The active `mod.rs` is 431 KB, well above the 500-line project guideline. This is a genuine maintenance hazard. Severity could be HIGH given the risk of version confusion. |
| 8 | LOW | Redundant second `cancel_adjacent_pairs` call | **REAL** | The backtrack `i -= 1` already handles cascading cancellations. The second pass finds nothing the first pass missed (it applies the same adjacent-only logic). This is technically redundant. For non-adjacent interleaved patterns, the second pass is also powerless, so it provides no safety benefit. Correct analysis. |
| 9 | LOW | `remove_identities` unconditional full copy | **REAL** | `barriers.to_vec()` copies the entire input before filtering. The suggested iterator alternative (`barriers.iter().filter(...).copied().collect()`) avoids the temporary allocation for discarded elements. Trivial micro-optimization. |

---

## Key Cross-Cutting Observation

The junior was biased toward flagging but **missed the single most important finding**: the entire BarrierOptimizer test suite does not compile. Neither `blackbox_barrier_opt.rs` (46 test functions) nor `whitebox_t_fg_9_5_regression.rs` integrates with the current `mod.rs`. Every call to `BarrierOptimizer::new()` (found in 60+ locations across test files) is a compile error. Every test passing a 5-tuple where a 6-tuple is expected is a compile error. Every reference to `compiled.stats.barriers_optimized` or `enable_barrier_opt` is a compile error. The "376 tests pass" claim in the junior's backdrop cannot refer to these test files.

---

## Summary

| # | Junior Severity | Senior Verdict | Adjusted Severity |
|---|----------------|----------------|-------------------|
| 1 | CRITICAL | Overzealous severity | HIGH |
| 2 | CRITICAL | REAL | CRITICAL |
| 3 | HIGH | Overzealous severity | MEDIUM |
| 4 | MEDIUM | REAL | MEDIUM |
| 5 | MEDIUM | REAL | MEDIUM (or LOW) |
| 6 | MEDIUM | REAL (moot) | MEDIUM |
| 7 | MEDIUM | REAL (understated) | MEDIUM (or HIGH) |
| 8 | LOW | REAL | LOW |
| 9 | LOW | REAL | LOW |

**Real findings**: 2 (CRITICAL), 4, 5, 6, 7, 8, 9 (7 of 9)
**Overzealous severity**: 1 (CRITICAL -> HIGH), 3 (HIGH -> MEDIUM)
**Missed by junior**: The test suite does not compile against the production code.
