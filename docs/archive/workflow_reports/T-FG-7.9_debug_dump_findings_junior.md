# T-FG-7.9 Junior QA Findings

**Reviewer:** Junior QA
**Scope:** DebugDumper, Display for CompiledFrameGraph, TRINITY_DUMP_FRAME_GRAPH env var
**Files reviewed:**
- `crates/renderer-backend/src/frame_graph/debug_dumper.rs` (542 lines)
- `crates/renderer-backend/src/frame_graph/mod.rs` (Display impl at line 3823, env-var gate at line 3496, struct + compile() at lines 3080-3501)

---

## Finding 1 -- Unused import of `core::fmt` [WARNING]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, line 18

The import `use core::fmt;` is unused in this file. The file only uses `format!`, `String::push_str`, and `String::push` -- none of which require importing `core::fmt`. The `Display` impl that uses `fmt::Formatter` lives in `mod.rs`, not here.

The compiler will emit a warning for this unused import, visible under `#[warn(unused_imports)]`.

**Fix:** Remove the line.

---

## Finding 2 -- `count_optimized_barriers` reports a meaningless metric [BUG]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, lines 526-541

The function computes:

```
scheduled_count - graph.barriers.len()
```

where `scheduled_count` is the total `(pre_barriers + post_barriers)` across all `scheduled_passes`, and `graph.barriers.len()` is the count of raw 4-tuple barriers.

The problem: these two numbers come from structurally different representations. `graph.barriers` are 4-tuples `(PassIndex, PassIndex, ResourceState, ResourceState)` produced directly by `compute_barriers()`. The `scheduled_passes` store 6-tuple `BarrierTuple` entries produced by `barriers_4tuple_to_barrier_tuples()`, which iterates over dependency *edges* and can produce a different number of entries than the raw barrier list (multiple edges between the same passes with different resources can map to differing counts).

The result displayed as "Barriers optimized" has no reliable meaning -- it can be non-zero even when `BarrierOptimizer` did nothing, or zero when it did remove barriers. The metric should be computed directly by comparing `optimized_barriers.len()` against `barrier_tuples.len()` before and after optimization.

**Severity:** Medium -- the label is misleading. A developer reading the dump may wrongly conclude barriers are being optimized when they are not, or vice versa.

**Fix:** Either (a) hold a separate `barriers_optimized_away` counter in `CompiledFrameGraph` set during `compile_with_config()`, or (b) rename the field to something honest like "Barriers (scheduled delta)".

---

## Finding 3 -- PERFORMANCE COUNTERS section has placeholders mixed with real data [CLARITY]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, lines 254-261

The PERFORMANCE COUNTERS section shows:

```
  DAG build             : -- us
  Topological sort      : -- us
  Barrier compute       : -- us
  Async schedule        : -- us
  Dead-pass elim        : -- us
  Total                 : 1234 us
```

Individual phase counters are hard-coded to `"-- us"` (placeholders). Only `Total` shows a real value from `compilation_time_us`. This creates a confusing visual where the breakdown rows don't sum to the total, and a user might think the timing infrastructure is broken.

**Severity:** Low -- clearly announced as scaffolding but prone to misinterpretation.

**Recommended action:** Either (a) add phase-level timing fields to `CullStats` or `CompiledFrameGraph` and wire them through the compiler pipeline, or (b) remove the individual rows and keep only Total until per-phase timers exist.

---

## Finding 4 -- No unit tests for DebugDumper output [TESTING GAP]

**File:** `crates/renderer-backend/src/frame_graph/` (entire module)

There are no tests for:
- `DebugDumper::dump()` output format or content
- `CompiledFrameGraph`'s `Display` impl
- `TRINITY_DUMP_FRAME_GRAPH` env-var gate

Existing `*_display` tests in `mod.rs` cover individual IR types (`CullStats`, `ResourceHandle`, `IrPass`, etc.) but none construct a `CompiledFrameGraph` and verify that `format!("{}", graph)` produces sensible output (e.g. contains expected section headers, pass names, resource handles).

**Severity:** Low-medium -- the output is for human debugging so correctness is less critical, but a regression test would prevent silent breakage.

**Recommended action:** Add a test that:
1. Creates a small `CompiledFrameGraph` with known passes/resources/edges
2. Calls `DebugDumper::dump()` or `format!("{}", graph)`
3. Asserts the output contains expected section markers and data values

---

## Finding 5 -- O(n*m) pass lookups for each barrier, edge, async pass, and eliminated pass [PERFORMANCE]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, multiple locations

The code repeatedly uses `graph.passes.iter().find(|p| &p.index == pass_idx)` to resolve a `PassIndex` to a pass name. Specifically:

- **Write barriers** (line 394-401): two `find()` calls per barrier
- **Write edges** (line 421-424): callers pass `passes` but `write_edge` doesn't use it (the parameter is accepted but unused -- the resource name lookup uses `resources`, not `passes`)
- **Async passes** (line 156-159): one `find()` per async pass
- **Eliminated passes** (line 201-206): one `find()` per eliminated pass
- **Pass execution order** (line 110-114): one `find()` per pass in order

For a frame graph with 500 passes and 2000 barriers, this results in approximately 4000+ linear scans. Building a `HashMap<PassIndex, &IrPass>` once at the top of `dump()` would reduce this to O(1) per lookup.

**Severity:** Low for debugging code (not on a hot path), but the perf impact is predictable at scale.

**Fix:** Build a pass-index map at the top of `dump()`:
```rust
let pass_map: HashMap<PassIndex, &IrPass> = graph.passes.iter().map(|p| (p.index, p)).collect();
```
Then pass it to helper functions instead of the full `&[IrPass]`.

---

## Finding 6 -- `write_edge` accepts but does not use `passes` parameter [CLEANLINESS]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, line 416

The `write_edge` function signature is:
```rust
fn write_edge(out, edge, passes, resources) { ... }
```

But the body only uses `resources` (to look up the resource name). The `passes` parameter is dead. Callers still pass `&graph.passes` making the call slightly misleading.

**Fix:** Remove the unused `passes` parameter, or add a pass-name column to the edge output (showing the `from`/`to` pass names alongside the indices).

---

## Finding 7 -- Orphan `PassIndex` values in `order` are silently skipped [EDGE CASE]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, lines 110-115

```rust
for pass_idx in &graph.order {
    if let Some(pass) = graph.passes.iter().find(|p| &p.index == pass_idx) {
        write_pass(&mut out, pass);
    }
}
```

If `graph.order` contains a `PassIndex` with no matching entry in `graph.passes` (e.g. from a manually constructed graph or a serialization round-trip error), the dump silently skips it with no indication. The same pattern applies to the eliminated-passes iterator (line 202) which uses `unwrap_or("<eliminated>")` -- that path is handled better.

**Fix:** Add an `else` branch:
```rust
} else {
    out.push_str(&format!("  P{} <missing pass definition>\n\n", pass_idx.0));
}
```

---

## Finding 8 -- Depth info is not displayed despite comment claiming otherwise [DOCUMENTATION]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, lines 351-353

```rust
// Depth if available is shown via the depths map at the call site.
// Here we show the pass details.
```

The `dump()` function never prints pass depths. The comment suggests depths will be displayed "at the call site" but no call site in `dump()` does this. The `graph.depths` `HashMap` exists and is populated (Phase 2c of the compiler), so the data is available.

**Severity:** Low (cosmetic/misleading comment).

**Fix:** Either display depth in `write_pass` output (e.g. `P3 "shadow_map" [Graphics] depth=2`) or remove the comment.

---

## Finding 9 -- Header does not account for dynamically skipped passes [ACCURACY]

**File:** `crates/renderer-backend/src/frame_graph/debug_dumper.rs`, lines 64-66

```rust
let alive = graph.order.len();
let eliminated = graph.cull_stats.passes_eliminated;
```

After `apply_runtime_culling()` is called, `graph.order` contains fewer entries (dynamically skipped passes are removed). The header then reads as, e.g.:

```
Passes (total)        : 15 (8 alive, 3 eliminated)
```

The user sees 8 + 3 = 11, not 15. The missing 4 are dynamically skipped, but the header never mentions this. The `dynamically_skipped` field only appears in the COMPILER STATISTICS section, making the header arithmetic confusing on its own.

**Fix:** Either (a) adjust the header to include dynamic skips, or (b) add a note that the order length reflects post-cull state.

---

## Summary

| # | Finding | Severity | Type |
|---|---------|----------|------|
| 1 | Unused import `use core::fmt;` | Warning | Cleanliness |
| 2 | `count_optimized_barriers` metric is structurally meaningless | Medium | Correctness |
| 3 | PERFORMANCE COUNTERS section has placeholder sub-rows | Low | Clarity |
| 4 | No unit tests for DebugDumper output | Low-Medium | Testing Gap |
| 5 | O(n*m) pass lookups via linear `find()` at scale | Low | Performance |
| 6 | Unused `passes` parameter in `write_edge` | Low | Cleanliness |
| 7 | Orphan PassIndex values silently skipped | Low | Edge Case |
| 8 | Misleading comment about depth display | Low | Documentation |
| 9 | Header arithmetic after dynamic culling is off | Low | Accuracy |

**Overall assessment:** The implementation is functionally correct and will produce useful debug output. The primary issue is finding 2 (meaningless optimization metric), which could mislead developers. Finding 1 (unused import) is a compiler warning. Findings 4-9 are minor. The code is structurally sound, handles the empty-case paths consistently (showing `(none)` for each section), and the env-var/integration in `compile_with_config()` is clean.
