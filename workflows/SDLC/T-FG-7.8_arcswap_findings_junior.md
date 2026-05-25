# T-FG-7.8 ArcSwap -- Junior QA Findings

**Reviewer**: Junior QA
**Review date**: 2026-05-23
**Files reviewed**:
- `crates/renderer-backend/src/frame_graph/swap.rs` (implementation + whitebox tests)
- `crates/renderer-backend/tests/blackbox_frame_graph_swap.rs` (blackbox tests)
- `crates/renderer-backend/src/frame_graph/mod.rs` (CompiledFrameGraph, mod declarations)

**Test count**: 392 lib tests pass (9 swap-specific whitebox + 383 existing). 22 blackbox integration tests exist for the swap module.

---

## Summary

`FrameGraphSlot` is a thin `ArcSwap<CompiledFrameGraph>` wrapper providing `new`, `load`, `store`, `swap`, `compile_and_swap`, and `is_loaded`. The implementation is functionally correct -- the core methods are one-liners that delegate directly to the well-tested `arc_swap` crate. The blackbox tests are thorough, covering single-threaded correctness, concurrent read/write, empty-graph edge cases, and Send+Sync trait bounds.

**Verdict**: PASS with minor findings. No correctness bugs. Three code-quality issues and three test-coverage gaps.

---

## Code Quality Findings

### CQ-1. `is_loaded()` is a constant stub (low severity)

`FrameGraphSlot::is_loaded()` unconditionally returns `true`. The doc comment on line 150 admits this explicitly: *"This is always true after construction via new, but provides a consistent API for future scenarios."*

**Impact**: Dead code in the current implementation. If a consumer ever expects `is_loaded()` to reflect a real state (e.g., the slot was cleared or not yet initialized), this will silently give a false positive.

**Recommendation**: Either (a) remove the method until it has real semantics, or (b) implement `Option<ArcSwap<...>>` internally and track a `None` state if lazy initialization is a planned feature. Currently it is misleading.

### CQ-2. `compile_and_swap` uses `store` not `swap` (low severity)

The method is named `compile_and_swap` but calls `self.store(...)` internally (line 143), not `self.swap(...)`. This means the old graph is silently dropped instead of returned to the caller.

```rust
pub fn compile_and_swap(&self, passes: Vec<IrPass>, resources: Vec<IrResource>) -> Result<(), String> {
    let new_graph = CompiledFrameGraph::compile(passes, resources)?;
    self.store(new_graph);  // <-- uses store, not swap
    Ok(())
}
```

**Impact**: The method name implies the caller could inspect the replaced graph (e.g., to compare compilation results, diagnose performance regressions, or keep it alive for deferred rendering). The current signature returns `()` and discards it.

**Recommendation**: Either rename to `compile_and_store` to match actual behaviour, or change the return type to `Result<Arc<CompiledFrameGraph>, String>` and use `self.swap(...)`. The latter is more consistent with the public API of `FrameGraphSlot`.

### CQ-3. `load()` always uses `load_full` (medium-low severity)

The `load()` method calls `self.current.load_full()` (line 104), which always performs an atomic increment on the Arc reference count. The `arc_swap` crate also provides `load()` which returns a `Guard` that borrows the slot's internal pointer without bumping the refcount -- cheaper for short-lived snapshots.

```rust
pub fn load(&self) -> Arc<CompiledFrameGraph> {
    self.current.load_full()  // Atomic increment every frame
}
```

**Impact**: In the hot render loop (called once per frame), this is an unnecessary atomic increment. While a single atomic increment is cheap, it compounds with every graph field access through the resulting Arc. If the render thread only needs the graph for the duration of a single function scope, `Guard`-based access eliminates the refcount traffic entirely.

**Recommendation**: Consider exposing both `load()` (returning `Arc<...>`, for long-lived references that must outlive the slot) and a `borrow()` method (returning `arc_swap::Guard<...>`, for per-frame read-only access). The doc comment already describes per-frame access as the primary use case -- `Guard` is the better default for that path.

---

## Test Coverage Gaps

### GAP-1. No `compile_and_swap` failure-recovery test (medium severity)

There is no test that verifies the slot's state when `compile_and_swap` receives invalid IR (passes with a dependency cycle, mismatched resource handles, etc.). The method on line 137-145 propagates compilation errors without modifying the slot, but this behaviour is untested.

**Existing test** (`test_compile_and_swap_succeeds`): Tests only the success path with valid inputs.

**Missing**: A test that submits (a) duplicate resource handles, (b) a cycle in the pass graph, or (c) mismatched attachments, and then asserts that `load()` still returns the previous graph unchanged.

### GAP-2. No memory-lease / Arc-count verification (low severity)

The concurrent tests verify that threads do not panic and that loaded data is consistent, but never check that Arcs are properly decoupled -- i.e., that dropping all handles to an old graph eventually frees it. This is guaranteed by Rust's Arc semantics, but a regression (e.g., an accidental `Arc::into_raw` without a matching `from_raw`) could cause a silent leak that no current test would catch.

**Recommendation**: Add a test that constructs a graph, takes N `load()` handles, drops the slot, drops all handles, and verifies there is no leak (e.g., via a `Drop` counter on a wrapper, or by observing that a weak reference upgraded in a cycle-dropped scenario).

### GAP-3. No stress test under high write frequency (low severity)

The concurrent tests use 2 writers x 10 writes (test 7a) and 1 writer x 20 writes (test 7b). `ArcSwap` uses an internal RCU-based slot mechanism where frequent writes (every frame) can cause old copies to accumulate until all readers have cycled. The doc comment dismisses this as negligible (line 16-17: *"recompilation happens orders of magnitude less often than per-frame reads"*), but there is no test that verifies memory does not grow unbounded under burst-write conditions (e.g., 100 rapid writes in succession).

**Recommendation**: Add a test that performs 1,000 back-to-back `store()` calls (no interleaved reads) and measures that the slot's retained memory stays bounded. This is a regression test against accidental changes that might pin old copies.

---

## Strengths

1. **Clean, minimal design.** The struct is a single field -- a thin wrapper over `ArcSwap`. Each pub method is one or two lines. No unnecessary abstraction.

2. **Blackbox tests are thorough.** The cleanroom test file covers 22 distinct scenarios including construction, load/store/swap, old-Arc validity after mutation, concurrent readers+writers, empty graph round-trips, multiple sequential mutations, and Debug formatting. Thread safety is exercised with real `std::thread::spawn` calls (not just compile-time trait bounds).

3. **Thread-safety verified at compile time and at runtime.** The whitebox `test_send_sync_bounds` uses a const generic assertion. The blackbox `slot_is_send` and `slot_is_sync` tests exercise real thread transfer and shared-borrow across threads.

4. **Safety comment is present and accurate.** Lines 157-160 explain why no unsafe impl is needed.

5. **Doc comments include an ASCII architecture diagram** (lines 46-52) showing the render thread / compilation thread relationship. Good for maintainability.

6. **No correctness bugs found.** The core operations (`load_full`, `store`, `swap`) delegate to `arc_swap` which is a mature, well-audited crate. The wrapper adds no incorrect logic.

---

## Test Inventory

### Whitebox (in `swap.rs`)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_send_sync_bounds` | Compile-time Send + Sync |
| 2 | `test_new_creates_loaded_slot` | `is_loaded()` after `new()` |
| 3 | `test_load_returns_stored_graph` | `load()` returns stored graph by name |
| 4 | `test_store_replaces_graph` | `store()` replaces the graph |
| 5 | `test_swap_returns_old_arc` | `swap()` returns previous `Arc` |
| 6 | `test_old_arc_remains_valid_after_swap` | Old Arc survives slot mutation |
| 7 | `test_multiple_loads_return_consistent_data` | Two `load()` calls match |
| 8 | `test_compile_and_swap_succeeds` | `compile_and_swap` with valid IR |
| 9 | `test_slot_with_minimal_graph` | Single-pass graph round-trip |

### Blackbox (in `tests/blackbox_frame_graph_swap.rs`)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `slot_constructed_with_compiled_graph` | Construction |
| 2 | `load_returns_arc_to_stored_graph` | `load()` dereferences to graph |
| 3 | `load_multiple_calls_produce_independent_arcs` | Multiple loads see same data |
| 4 | `store_replaces_graph_atomically` | `store()` changes graph data |
| 5 | `store_preserves_outstanding_references` | Old Arc survives `store()` |
| 6 | `swap_returns_old_graph` | `swap()` return value |
| 7 | `swap_arc_remains_valid_after_further_swaps` | Chain of 3 swaps preserves all old Arcs |
| 8 | `old_load_arc_still_valid_after_swap` | Pre-swap `load()` Arc survives |
| 9 | `swap_returned_arc_still_valid_after_another_swap` | Swap return Arc survives another swap |
| 10 | `is_loaded_returns_true_after_construction` | `is_loaded()` post-construction |
| 11 | `is_loaded_persists_through_operations` | `is_loaded()` after store/swap |
| 12 | `concurrent_read_write_no_panic` | 4 readers x 200 + 2 writers x 10 |
| 13 | `concurrent_readers_with_store_do_not_panic` | 8 readers x 100 + 1 writer x 20 |
| 14 | `empty_graph_storage_works` | Empty graph in slot |
| 15 | `store_empty_over_non_empty_works` | Replace 3-pass with empty |
| 16 | `store_non_empty_over_empty_works` | Replace empty with 4-pass |
| 17 | `multiple_stores_in_sequence` | 3 sequential `store()` calls |
| 18 | `multiple_swaps_in_sequence` | 3 sequential `swap()` calls + old Arc check |
| 19 | `many_swaps_do_not_degrade_correctness` | 20 store/verify iterations |
| 20 | `debug_format_does_not_panic` | `Debug` does not panic |
| 21 | `slot_is_send` | Runtime Send via thread::spawn(move) |
| 22 | `slot_is_sync` | Runtime Sync via Arc + shared refs |

---

## Verdict

**PASS** -- No blocking issues. The implementation is correct, well-documented, and backed by strong test coverage (31 tests across whitebox and blackbox layers). The three code-quality findings (CQ-1, CQ-2, CQ-3) are low-severity polish items. The three coverage gaps (GAP-1, GAP-2, GAP-3) would strengthen confidence but are not release-blocking for the current feature scope.

**Recommendations for SENIOR_QA / FINAL sign-off**:
- Address CQ-2 (`compile_and_swap` vs `compile_and_store` naming) before merge
- Consider GAP-1 (compile_and_swap failure recovery) for the hardening pass
- CQ-3 (`load()` vs `borrow()`) is an optimization -- file as follow-up if per-frame perf matters
