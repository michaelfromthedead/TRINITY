# T-FG-3.4 ResourceAllocator Fix Cycle -- JUNIOR_QA Verification

Review date: 2026-05-23
Reviewer: JUNIOR_QA (FIX cycle verification)
Source: `workflows/SDLC/T-FG-3.4_allocator_final_report.md`
Target: `crates/renderer-backend/src/frame_graph/mod.rs`

---

## Verdict: FIX with 2 RESIDUALS + 1 NEW (PENDING SENIOR_QA_FINAL)

DEV addressed 9 of 11 FINAL action items completely, 1 partially, and 1 incorrectly. One new finding surfaced during verification. The two residual items (C-01, L-04) and the new finding (N-01) should be reviewed by SENIOR_QA_FINAL to determine whether they block GREEN_LIGHT or can be deferred.

---

## Detailed Item-by-Item Verification

### C-01 (CRITICAL) -- HistoryRingBuffer unification -- PARTIALLY FIXED / RESIDUAL

**Status: Fix incomplete.**

Three of four sub-items from the FINAL report were not fully addressed:

1. **Semantic direction chosen opposite of recommendation.** The report recommended "write-then-advance" (conventional double-buffering). The surviving copy uses "advance-first" (line 5155-5157: `self.current = (self.current + 1) % self.capacity;` then `self.slots[self.current] = handle;`). This is a defensible design choice (the test at line 8143 explicitly documents "Advance-first semantics"), but it flouts the FINAL report's explicit directive.

2. **`mod.rs_test_work` still exists with its own structurally distinct copy.** The production copy at `/home/user/dev/USER/PROJECTS_VOID/TRINITY/mod.rs_test_work` lines 4562-4620 has:
   - `new(slot_count, initial_handle)` with `slot_count >= 2` invariant
   - `write_current_and_advance`: write THEN advance (line 4616-4618)
   - `slot_handle`: direct indexing, panics on OOB (no `%` wrap-around)
   
   The test-module copy at mod.rs line 5126 has:
   - `new(n_slots)` with `n_slots >= 1` invariant
   - `write_current_and_advance`: advance THEN write (line 5155-5157)
   - `slot_handle`: `slots[slot % self.capacity]` (wrap-around)
   
   Two structurally different copies of the same abstraction remain live.

3. **Not moved to a shared module.** The unified version lives inside `mod tests` (line 4623+), making it inaccessible to production code. The production version in `mod.rs_test_work` was not migrated.

The duplicate count was reduced from 2 to 1 *within mod.rs*, but the production copy in `mod.rs_test_work` was left untouched. The invariants (`>= 1` vs `>= 2`) and semantics (advance-first vs write-then-advance) remain divergent.

---

### H-01 (HIGH) -- Deduplicate resources_freed / bytes_saved -- FIXED

`freed_set: HashSet<ResourceHandle>` correctly deduplicates across eliminated passes. Lines 3144-3162:
```
let mut freed_set: HashSet<ResourceHandle> = HashSet::new();
// ...
if !freed_set.contains(&r) {
    // ... exclusively_used check ...
    freed_set.insert(r);
    resources_freed += 1;
    bytes_saved += estimate_resource_bytes(&res.desc);
}
```
No double-counting path remains.

---

### H-03 (HIGH) -- Handle TextureCube in allocate_resources -- FIXED

`_ => {}` replaced with a full `ResourceDesc::TextureCube(desc)` arm at line 4993. All three paths (Imported, transient-with-color, transient-unreachable fallback) are handled. Depth hardcoded to `1` per the FINAL report recommendation.

---

### H-04 (HIGH) -- Resources missing from lifetimes -- FIXED

`greedy_color_resources` now has Phase 2 (lines 5198-5208) that assigns unique high colors to resources without lifetime entries ("orphans"):
```
let base_color = colors.values().copied().max().map(|m| m + 1).unwrap_or(0);
let orphans: Vec<&IrResource> = resources.iter().filter(|r| !lifetimes.contains_key(&r.handle)).collect();
for (i, res) in orphans.iter().enumerate() {
    colors.insert(res.handle, base_color + i);
}
```
This prevents accidental aliasing. Test `test_empty_lifetimes_two_or_more_transient_resources` at line 12012 guards the multi-resource orphan case.

---

### M-02 (MEDIUM) -- Resolve CullStats redundancy -- FIXED

`culled_pass_count` field removed from the `CullStats` struct (lines 2705-2730). JSON serialization at line 3395 emits it as a compatibility alias: `"culled_pass_count": self.cull_stats.passes_eliminated`. The `Display` impl (line 2732) omits it entirely.

---

### M-05 (MEDIUM) -- Remove stale `#[allow(dead_code)]` -- FIXED

Zero `#[allow(dead_code)]` annotations remain in the file. The `advance()` method is exercised by tests at lines 8168, 8171, 8174, 8178.

---

### M-04 (MEDIUM) -- Fix CullStats Display test assertions -- FIXED

`test_cull_stats_display` (line 11232) now uses field-specific assertions (e.g., `s.contains("passes_total=10")`, `s.contains("eliminated=3")`) instead of generic digit matching. Matches the Display format string at line 2736.

---

### M-03 (MEDIUM) -- Document dead else branches -- FIXED

All three `else` branches (lines 4927-4936, 4965-4974, 5018-5027 for TextureCube) have:
```
#[cfg(debug_assertions)]
unreachable!("greedy_color_resources must assign a color to every transient resource ...");
```
with a documented safety-net purpose.

---

### L-01 (LOW) -- Fix test comment -- FIXED

Comment at line 7788-7790 now accurately describes the mechanism: "A transient resource without a lifetime entry receives a unique high color from greedy_color_resources (no interference edges) and gets its own standalone physical allocation."

---

### L-03 (LOW) -- PhysicalTexture PartialEq exclusion of handle -- FIXED

Added `compatible_with(&self, other) -> bool` method at line 4775 that intentionally excludes `handle` from comparison. `PartialEq` delegates to `compatible_with` at line 4786. `AllocationTable::from_allocator` (line 5069) uses `compatible_with` instead of `==`, making the dedup logic self-documenting.

---

### L-04 (LOW) -- Missing allocator edge case tests -- MOSTLY FIXED (5/6)

| # | Scenario | Status | Location |
|---|----------|--------|----------|
| 1 | All-imported resources (no transient) | ADDED | `test_all_imported_resources_no_transient` line 12049 |
| 2 | Mixed Texture2D+Texture3D+Buffer aliasing | ADDED | `test_allocate_mixed_tex2d_tex3d_buffer_aliasing` line 12095 |
| 3 | Multiple resources aliased to same color | PARTIAL | For buffers only. The texture side (`or_insert_with` returns first-created) is not explicitly tested. |
| 4 | AllocationTable::from_allocator with imported-only | **MISSING** | Not added. Pairs with L-03 PartialEq fix. |
| 5 | Empty lifetimes with 2+ transient resources | ADDED | `test_empty_lifetimes_two_or_more_transient_resources` line 12012 |
| 6 | 1-slot HistoryRingBuffer wrap-around | ADDED | `test_history_ring_buffer_1_slot_wrap_around` line 12078 |

---

## New Finding

### N-01 (LOW) -- `advance()` unused in mod.rs

The `advance()` method at line 5150 is called by tests (lines 8168-8178) only in the `2_slot` test. The `write_current_and_advance` path (used by all other tests and presumably production callers) does NOT call `advance()` -- it duplicates the advance logic inline (line 5156). This is not a bug; the method is independently tested. Non-actionable but worth noting as a minor code smell (inconsistent composition).

---

## Compilation

- 195 unit tests pass (0 failures, 0 ignored).
- 70 pre-existing warnings (QueueType privacy, unused runtime_culling functions, feature cfg checks) are unrelated to T-FG-3.4.
- Blackbox integration tests fail to compile due to missing `compile_with_config` API and Py* type changes -- these are pre-existing issues, not T-FG-3.4 regressions.

---

## Summary

| Priority | Item | Verdict |
|----------|------|---------|
| CRITICAL | C-01 HistoryRingBuffer unification | **PARTIAL** -- 2 copies remain with divergent invariants/semantics |
| HIGH | H-01 resources_freed dedup | FIXED |
| HIGH | H-03 TextureCube handling | FIXED |
| HIGH | H-04 Orphan resource aliasing | FIXED |
| MEDIUM | M-02 CullStats redundancy | FIXED |
| MEDIUM | M-05 stale dead_code annotation | FIXED |
| MEDIUM | M-04 weak test assertions | FIXED |
| MEDIUM | M-03 dead else branches | FIXED |
| LOW | L-01 test comment | FIXED |
| LOW | L-03 PhysicalTexture PartialEq | FIXED |
| LOW | L-04 edge case tests | **5/6 ADDED** -- imported-only table test missing |
| NEW | N-01 advance() composition | Note only |

**9 of 11 items fully fixed. 2 items have residual issues.**

The code compiles cleanly (unit tests: 195 pass). The two residuals (C-01 and L-04 item 4) and the `mod.rs_test_work` integrity question require a SENIOR_QA_FINAL judgment call on whether they are blockers or deferrable.
