# T-FG-3.4 ResourceAllocator / T-FG-6.5/7.5/3.5 Fixes -- JUNIOR_QA Findings

Review date: 2026-05-23
Reviewer: JUNIOR_QA
Target: `crates/renderer-backend/src/frame_graph/mod.rs` (test module) + `temp_edit.rs` (production stubs)

---

## CRITICAL

### C-01: HistoryRingBuffer semantic inversion between mod.rs and temp_edit.rs

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 4766-4768 vs `crates/renderer-backend/src/frame_graph/temp_edit.rs` lines 4485-4488

**Issue:** The two files implement `write_current_and_advance` with **opposite** execution order.

`mod.rs` (advance-first):
```rust
fn write_current_and_advance(&mut self, handle: ResourceHandle) {
    self.current = (self.current + 1) % self.capacity;  // advance first
    self.slots[self.current] = handle;                   // write to new slot
}
```

`temp_edit.rs` (write-then-advance):
```rust
pub fn write_current_and_advance(&mut self, handle: ResourceHandle) {
    self.slots[self.current] = handle;  // write to current
    self.advance();                      // then advance
}
```

The naming suggests "write then advance", which matches `temp_edit.rs`. The `mod.rs` version should be named `advance_and_write` or similar. If any production code is compiled against the wrong version, history slot reads return stale or pre-initialized handles. A runtime consumer reading `current_slot()` to determine "which frame's history am I reading" gets a one-slot-ahead pointer.

Similarly, the `slot_handle` method wraps with `%` in `mod.rs` but uses direct indexing in `temp_edit.rs`, meaning callers that pass an un-wrapped slot number work in one file but panic out-of-bounds in the other.

---

## HIGH

### H-01: `resources_freed` and `bytes_saved` double-count shared resources across eliminated passes

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 2979-2989

**Issue:** When multiple eliminated passes write the **same** resource, and no live pass reads it, the counting loop iterates per-pass and increments `resources_freed` / `bytes_saved` once per pass-write rather than once per unique resource. This inflates both statistics.

**Example:** Two eliminated compute passes both write `ResourceHandle(5)`. No other pass reads it. Loop iteration 1 counts it freed and adds its bytes. Loop iteration 2 counts it freed *again* and adds its bytes *again*. The `CullStats` says 2 resources freed when only 1 unique resource was actually reclaimed.

**Impact:** Statistics reported via `emit_*` are unreliable for capacity planning or debugging. The `bytes_saved` field feeds into allocator heuristics and could produce misleading pressure estimates.

**Suggested fix:** Track freed resources with a `HashSet<ResourceHandle>` to deduplicate before counting.

### H-02: `write_current_and_advance` name does not match implementation in mod.rs

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` line 4766

**Issue:** The method is named `write_current_and_advance` but **advances first, then writes to the new current slot**. Every caller and every reader naturally expects the name to describe the operation order. This is a correctness trap for any future developer who uses the method without reading the full implementation.

**Impact:** Silent data corruption in history buffers. If someone ports `temp_edit.rs` production code that calls `write_current_and_advance` expecting write-then-advance, the aliasing test at line 7769 (mod.rs) passes because tests match their local implementation, but production behavior differs.

### H-03: `allocate_resources` silently drops unknown `ResourceDesc` variants

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` line 4640

**Issue:** The `match &res.desc` block handles `Texture2D`, `Texture3D`, and `Buffer`, but has a wildcard `_ => {}` that silently discards any other resource type. If a new variant is added (e.g., `TextureCube`, `AccelerationStructure`, `Sampler`), it compiles without error but produces zero allocations for that resource.

**Impact:** Future resource types will fail silently. A downstream pass that depends on the resource gets an uninitialized handle at runtime rather than a compile-time error.

**Suggested fix:** Either add a `todo!()` or `compile_error!` for unhandled variants, or exhaustively match.

### H-04: `InterferenceGraph::build` ignores resources not in `lifetimes`, creating incorrect aliasing

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 2207-2251

**Issue:** `InterferenceGraph::build` only considers resources present in the `lifetimes` map. Resources in the IR but absent from `lifetimes` are invisible to the graph. Consequently, `greedy_color_resources` assigns them color 0 with no neighbors, which allows the allocator to alias them with any other color-0 resource -- even if they have overlapping lifetimes (the lifetime information is simply not in the graph).

The test at line 7379 (`test_allocate_resource_handle_not_in_lifetimes`) explicitly tests this path with a single resource and empty lifetimes, which "works" but only by coincidence -- one resource with color 0 aliases to itself. With two such resources the aliasing is wrong.

**Impact:** Transient resources accidentally omitted from `lifetimes` silently share physical memory with unrelated resources, producing read-after-write hazards on the GPU.

---

## MEDIUM

### M-01: `greedy_color_resources` assigns colors to imported and history resources unnecessarily

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 4778-4803

**Issue:** The coloring loop iterates **all** resources, including those with `ResourceLifetime::Imported`. The caller (`allocate_resources`) never uses the color for imported resources -- they always get a unique allocation. The coloring work is wasted CPU time.

For a frame graph with hundreds of imported resources, this is quadratic complexity in the interference graph on resources that are never aliased.

**Impact:** Performance regression on large frame graphs. Minor in isolation (JSON round-trip tests), but cumulative in a real rendering loop.

### M-02: CullStats redundancy: `passes_eliminated` and `culled_pass_count` are always equal

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 2650/2658, 3000/3004

**Issue:** Both fields are initialized from the same expression (`eliminated.len()`) at the single construction site. They are guaranteed to be identical by construction. This is dead weight in the struct, increases serialized JSON size, and is a maintenance liability (future changes must remember to keep both in sync).

**Impact:** Cleanliness. Could confuse downstream tooling that expects `culled_pass_count` to measure something different from `passes_eliminated` (e.g., a later filtering step).

### M-03: `else` branches in `allocate_resources` are dead code

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 4579-4588, 4616-4624, 4635-4637

**Issue:** Each resource type's match arm has an `else` branch for when `colors.get(&res.handle)` returns `None`. But `greedy_color_resources` assigns a color to **every** resource in the input slice, so this path is unreachable. It looks like a leftover from an earlier design where transient resources without a lifetime entry were not colored.

**Impact:** Dead code with no test coverage. If someone refactors `greedy_color_resources` to skip certain resources, the fallback path may silently produce standalone allocations that defeat aliasing -- with no test to catch the regression.

### M-04: CullStats Display test uses value assertions that match multiple fields

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 10822-10840

**Issue:** `test_cull_stats_display` asserts `s.contains("3")`, which matches **both** `passes_eliminated=3` and `culled_pass_count=3`. If one of these fields were removed or changed to a different value, the test would still pass as long as at least one "3" appears in the output. The test also does not assert for the field labels themselves (e.g., `s.contains("passes_total=")`), making it a weak oracle.

**Impact:** A regression that silently drops or mislabels a field from the Display output would not be caught.

### M-05: `advance()` has `#[allow(dead_code)]` annotation inside test module

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` line 4760

**Issue:** The `HistoryRingBuffer` struct and all its methods live inside `mod tests { ... }` (start at line 4305). Since `advance` is called in `test_history_ring_buffer_2_slot_matches_double_buffering` (line 7759), it IS used within its module. The `#[allow(dead_code)]` annotation is a dead annotation itself -- the Rust compiler would not warn on it since it is referenced within the same module.

**Impact:** Trivial, but indicates the struct may have been moved into the test module from production code without cleaning up the annotation. Creates confusion about the intended scope of the type.

---

## LOW

### L-01: Test comment misstates fallback when resource is not in lifetimes

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 7379-7381

**Text:** "A transient resource without a lifetime entry falls back to (PassIndex(0), PassIndex(0)) -- still allocates correctly."

**Issue:** The `lifetimes` parameter is an empty `HashMap`, not a map containing `(PassIndex(0), PassIndex(0))`. The comment describes an expected fallback behavior that is not implemented in the code. The resource "works" only because `InterferenceGraph::build` with an empty lifetimes map produces no edges, and `allocate_resources` does not itself read the lifetimes map for any per-resource decisions. The comment describes a non-existent mechanism.

### L-02: `num_colors` returns 0 for empty map instead of proper zero color count

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` line 4805-4807

**Issue:** `num_colors` returns `max().map(|m| m + 1).unwrap_or(0)`. When the map is empty, it correctly returns 0. But when the map has entries starting from color 1 (no color 0), it returns 1 instead of 2. While this scenario is unlikely with greedy coloring (which always starts from 0), it is a brittle implementation. Prefer `colors.values().max().copied().unwrap_or(0).saturating_add(1)` with explicit handling.

### L-03: `PhysicalTexture` partial_eq excludes `handle` field, allowing accidental dedup of imported resources

**File:** `crates/renderer-backend/src/frame_graph/mod.rs` lines 4489-4497

**Issue:** `PartialEq` for `PhysicalTexture` compares format, width, height, depth, and `is_transient` but **not** the `handle` field. In `AllocationTable::from_allocator`, two imported textures with matching dimensions/formats/transient=false are treated as the same physical resource and deduplicated to a single index. While imported textures are allocated uniquely in `allocate_resources`, the allocation table collapses them, which could confuse resource binding code that uses the physical index to reference distinct external resources.

This is currently latent because imported resources do not flow through the `AllocationTable` path in production (they are only used in test code), but any future production use of `from_allocator` with imported resources will silently alias them.

### L-04: Missing tests for allocator edge cases

**Files:** `crates/renderer-backend/src/frame_graph/mod.rs` test module

The following scenarios have no test coverage:

| Scenario | Risk |
|----------|------|
| `allocate_resources` with all-imported (no transient) resources | Verifies imported path works without transient interference |
| `allocate_resources` with mixed Texture2D + Texture3D + Buffer aliasing | Cross-type aliasing correctness |
| Multiple resources of different sizes aliased to the same color | Verifies the `or_insert_with` returns first-created physical resource |
| `AllocationTable::from_allocator` with imported-only resources | Imported dedup behavior (see L-03) |
| Empty `lifetimes` map with **multiple** transient resources | Interference graph empty => all alias to same color (see H-04) |
| 1-slot HistoryRingBuffer wrap-around | `n=1` is a degenerate case; `slot_handle` with `% 1` always returns slot 0 |

---

## Summary

| Severity | Count | Key Concerns |
|----------|-------|-------------|
| CRITICAL | 1 | HistoryRingBuffer semantic mismatch between test and production stubs |
| HIGH | 4 | Double-counted cull stats; reverse-named mutator; silent variant drop; broken aliasing for un-lifetimed resources |
| MEDIUM | 5 | Unnecessary work; redundant fields; dead branches; weak test assertions; stale annotation |
| LOW | 4 | Misleading comment; brittle helper; imported-resource dedup; coverage gaps |

The most actionable finding is **C-01**: the two copies of `HistoryRingBuffer` disagree on fundamental behavior (advance-first vs write-then-advance, `%` wrap vs direct index, `>=1` vs `>=2` slot minimum). Before this code can be promoted to production, these two versions must be unified and the semantics clearly documented at the type level.
