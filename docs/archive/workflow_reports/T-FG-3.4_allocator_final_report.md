# T-FG-3.4 ResourceAllocator / T-FG-6.5/7.5/3.5 Fixes -- SENIOR_QA_FINAL Report

Review date: 2026-05-23
Reviewer: SENIOR_QA_FINAL
Source: `workflows/SDLC/T-FG-3.4_allocator_sanity_report.md`
Target: `crates/renderer-backend/src/frame_graph/mod.rs` + `mod.rs_test_work`

---

## VERDICT: FIX

---

## Rationale

The implementation is architecturally sound. The `ResourceAllocator`, `InterferenceGraph`, greedy coloring algorithm, and the full allocation pipeline (dead-pass elimination, lifetime computation, barrier scheduling) are correctly designed and tested. The real bugs found are localized and each has a clear, well-scoped fix. No fundamental architectural flaws exist that would require a rewrite, and no blockers exist that would require escalation. However, one CRITICAL cross-type semantic mismatch (C-01) and three HIGH correctness bugs (H-01, H-03, H-04) prevent a GREEN_LIGHT verdict -- the code as-is would produce incorrect results in production.

---

## Consolidated Action Items for DEV Re-Entry

Items are ordered by severity. Each includes the file path, line reference, and recommended fix approach.

### ITEM 1 (CRITICAL): C-01 -- Unify HistoryRingBuffer semantics

**Files:**
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 5068-5106 (test module, private struct)
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/mod.rs_test_work` lines 4562-4619 (production struct)

**Problem:** Two copies of `HistoryRingBuffer` exist with incompatible contracts:

| Aspect | mod.rs (test, private) | mod.rs_test_work (production, public) |
|--------|------------------------|---------------------------------------|
| `write_current_and_advance` | advance first, then write to new slot | write to current slot, then advance |
| `slot_handle` | `slots[slot % self.capacity]` | `slots[slot_index]` (direct, panics on OOB) |
| `new` invariant | `n_slots >= 1` | `slot_count >= 2` |
| `new` signature | `(n_slots: usize)` | `(slot_count: usize, initial_handle: ResourceHandle)` |

The test in `mod.rs` at line 8066 documents "advance-first semantics," confirming deliberate divergence. Unification will silently break whichever caller expects the other contract.

**Action:**
1. Choose one semantic direction. Write-then-advance (like the `mod.rs_test_work` version) is the conventional double-buffering pattern -- the term "write current and advance" literally means "write the current slot, then advance."
2. Delete the duplicate in `mod.rs` at lines 5068-5106. The test will need updating to match the unified semantics.
3. Move the unified production version from `mod.rs_test_work` into a shared module accessible to both production and test code.
4. Verify all callers and update `slot_handle` call sites that rely on wrap-around (`%`).

### ITEM 2 (HIGH): H-01 -- Deduplicate resources_freed / bytes_saved counting

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 3146-3159

**Problem:** `resources_freed` and `bytes_saved` iterate over `eliminated_indices` (per eliminated pass), then per-pass writes. If two eliminated passes both write the same resource `R`, each pass independently increments `resources_freed` and `bytes_saved` for `R`, even though only one unique resource is reclaimed. The `exclusively_used` check at line 3152 only verifies that no *live* pass reads the resource -- it does not deduplicate across eliminated passes.

**Proof:** Loop at line 3149: `for &ei in &eliminated_indices` with inner `for &r in &passes[ei].access_set.writes`. If pass A (eliminated) writes R and pass B (eliminated) writes R, R is counted twice.

**Action:** Use a `HashSet<ResourceHandle>` to track already-counted resources. Replace lines 3146-3159 with:
```rust
let mut freed_resources: HashSet<ResourceHandle> = HashSet::new();
// ... inside loop ...
if exclusively_used && freed_resources.insert(r) {
    resources_freed += 1;
    bytes_saved += estimate_resource_bytes(&res.desc);
}
```

### ITEM 3 (HIGH): H-03 -- Handle TextureCube in allocate_resources

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` line 4972

**Problem:** `ResourceDesc` (line 789) has four variants: `Texture2D`, `Texture3D`, `Buffer`, and `TextureCube`. The match in `ResourceAllocator::allocate_resources` (lines 4886-4973) handles `Texture2D` (line 4887), `Texture3D` (line 4923), and `Buffer` (line 4959), then `_ => {}` (line 4972) silently drops `TextureCube`. Any `TextureCube` resource produces no allocation -- downstream passes reading it get an uninitialized handle.

**Confirmed:** `TextureCube` variant exists at line 537 and 795. It is used in `texture_format` (line 2237) and `estimated_bytes` (line 834), confirming it is a valid, live variant.

**Action:** Either:
- Replace `_ => {}` with an explicit `TextureCube` arm that handles it (same structure as `Texture2D` but with `desc.depth = 1`), or
- Replace `_ => {}` with `ResourceDesc::TextureCube(desc) => { unimplemented!("TextureCube allocation") }` if the feature is not yet supported (with tracking ticket reference).

### ITEM 4 (HIGH): H-04 -- Handle resources missing from lifetimes in InterferenceGraph

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 2260-2310 (build), 4875-4880 (call site), 5110-5135 (greedy_color_resources)

**Problem:** `InterferenceGraph::build` (line 2264, documented) only includes resources present in `lifetimes`. `greedy_color_resources` (line 5110) processes ALL resources in the input slice without checking whether they appear in the interference graph. Resources not in `lifetimes` get zero neighbors in `ig.neighbors(res.handle)`, so they receive color 0. In `allocate_resources`, the `colors.get()` branch activates (since every resource gets a color), and two resources with no lifetimes and overlapping use patterns silently alias -- producing GPU read-after-write hazards.

**Proof:** `greedy_color_resources` lines 5122-5127: `for neighbor in ig.neighbors(res.handle)` returns empty vec for resources not in the graph, so `used_colors` is empty, and `color = 0` is always assigned. The existing test `test_allocate_resource_handle_not_in_lifetimes` (line 7711) tests a single resource, masking the aliasing bug.

**Action:** Two options (choose one):
- **Defensive**: Before assigning color 0, check whether the resource has a lifetime entry. If not, assign a unique color for each such resource (or skip color 0). This prevents accidental aliasing.
- **Documented**: Update the test suite to explicitly test the multi-resource case, and document in `greedy_color_resources` that resources without lifetime entries receive color 0 and therefore alias. Add a compile-time or debug assertion that warns when this occurs.

### ITEM 5 (MEDIUM): M-02 -- Resolve CullStats redundancy

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 2716-2717, 3174

**Problem:** `culled_pass_count` is always initialized from the same value as `passes_eliminated` (line 3174 vs 3170). The struct comment at line 2717 says it is an alias. This wastes JSON serialization space and creates a maintenance liability.

**Action:** Either:
- Remove `culled_pass_count` entirely and update callers to use `passes_eliminated`, or
- Add a defensive assertion: `debug_assert_eq!(self.culled_pass_count, self.passes_eliminated);` in any method that reads both.

### ITEM 6 (MEDIUM): M-05 -- Remove stale `#[allow(dead_code)]`

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` line 5092

**Problem:** `advance()` at line 5093 is called at lines 8091, 8094, 8097 within `mod tests`, so the Rust compiler never emits a dead-code warning. The `#[allow(dead_code)]` annotation is stale.

**Action:** Remove `#[allow(dead_code)]` at line 5092. If the unified `HistoryRingBuffer` (see Item 1) has `advance` that is used in tests, it does not need the annotation.

### ITEM 7 (MEDIUM): M-04 -- Fix CullStats Display test assertions

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 11155-11173

**Problem:** `test_cull_stats_display` uses `s.contains("3")` which matches both `passes_eliminated=3` and `culled_pass_count=3`. If one field were removed or renamed, the test would still pass as long as some "3" remained in the output. No assertions check field labels.

**Action:** Add field-label assertions:
```rust
assert!(s.contains("passes_eliminated=3"));
assert!(s.contains("culled_pass_count=3"));
```
Or match against the full format string.

### ITEM 8 (MEDIUM): M-03 -- Document dead else branches

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 4911-4919, 4947-4956, 4967-4969

**Problem:** The `else` branches in `allocate_resources` (fired when `colors.get()` returns `None`) are unreachable because `greedy_color_resources` assigns a color to every resource. They serve as an untested safety net for future refactoring.

**Action:** Add a `#[cfg(debug_assertions)] unreachable!()` in each `else` branch so that if someone refactors `greedy_color_resources` to skip certain resources, the debug build catches it immediately. Document the purpose as "safety net for future refactoring -- if this breaks, greedy_color_resources is now returning None for active resources."

### ITEM 9 (LOW): L-01 -- Fix test comment

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 7712-7713

**Problem:** Comment reads "falls back to (PassIndex(0), PassIndex(0)) -- still allocates correctly" but there is no fallback to `(PassIndex(0), PassIndex(0))`. The resource works because `InterferenceGraph::build` with empty lifetimes produces no edges, `greedy_color_resources` gives it color 0, and the color path creates a standalone aliasing entry.

**Action:** Update comment to accurately describe the mechanism: "Resource with no lifetime entry receives color 0 from greedy_color_resources (no interference edges) and aliases with any other color-0 resource. Single-resource case: assigns a standalone physical allocation."

### ITEM 10 (LOW): L-03 -- Address PhysicalTexture PartialEq exclusion of handle

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` lines 4765-4772

**Problem:** `PartialEq` for `PhysicalTexture` excludes `handle`, so two textures with matching format/width/height/depth/is_transient but different handles are considered equal. In `AllocationTable::from_allocator` (line 5011), textures are deduplicated by `==`. Two imported textures with different handles but matching dimensions collapse to a single physical index.

**Action:** Either:
- Include `handle` in `PartialEq`, or
- Add a distinct `compatible_with(&self, other: &Self) -> bool` method for dedup logic that intentionally ignores handle, and use it in `from_allocator` instead of `==`. Document why handle is excluded from dedup comparison.

### ITEM 11 (LOW): L-04 -- Add missing allocator edge case tests

**File:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs` test module

**Missing scenarios:**
1. `allocate_resources` with all-imported resources (no transient)
2. Mixed Texture2D + Texture3D + Buffer aliasing (cross-type)
3. Multiple resources of different sizes aliased to same color (validates `or_insert_with` returns first-created)
4. `AllocationTable::from_allocator` with imported-only resources (pairs with L-03)
5. Empty `lifetimes` with two or more transient resources (exposes H-04 aliasing bug)
6. 1-slot HistoryRingBuffer wrap-around (degenerate case of `% 1` always returning 0)

Each is a one-test-function addition. Scenario 5 is the most important -- it directly guards against the H-04 bug.

---

## Summary

| Priority | Items | Status |
|----------|-------|--------|
| CRITICAL | 1 | HistoryRingBuffer unification |
| HIGH | 3 | Double-counting, TextureCube, aliasing from missing lifetimes |
| MEDIUM | 3 | CullStats redundancy, stale annotation, weak test assertions |
| LOW | 3 | Docs, PartialEq, coverage gaps |

**Total actionable items: 11**

All 11 items have clear, scoped fixes. No architectural redesign is needed. The code should return to DEV for a single fix pass, then proceed to SENIOR_QA_FINAL re-review.

---

## Verdict History

| Stage | Verdict | Date |
|-------|---------|------|
| JUNIOR_QA | 14 findings (1 CRITICAL, 4 HIGH, 5 MEDIUM, 4 LOW) | 2026-05-23 |
| SENIOR_QA_SANITY | 11 REAL, 3 OVERZEALOUS | 2026-05-23 |
| **SENIOR_QA_FINAL** | **FIX** -- 11 consolidated items | **2026-05-23** |
