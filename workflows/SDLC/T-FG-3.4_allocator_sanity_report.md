# T-FG-3.4 ResourceAllocator / T-FG-6.5/7.5/3.5 Fixes -- SENIOR_QA_SANITY Report

Review date: 2026-05-23
Reviewer: SENIOR_QA_SANITY
Source: `workflows/SDLC/T-FG-3.4_allocator_findings_junior.md`
Target: `crates/renderer-backend/src/frame_graph/mod.rs` + `temp_edit.rs`

---

## Summary

| Severity | JUNIOR Count | REAL | OVERZEALOUS |
|----------|-------------|------|-------------|
| CRITICAL | 1 | 1 | 0 |
| HIGH | 4 | 3 | 1 |
| MEDIUM | 5 | 4 | 1 |
| LOW | 4 | 3 | 1 |
| **Total** | **14** | **11** | **3** |

---

## CRITICAL

### C-01: HistoryRingBuffer semantic inversion between mod.rs and temp_edit.rs

**VERDICT: REAL**

The two files define the same named type with genuinely different semantics:

| Aspect | mod.rs (test module, line 4861) | temp_edit.rs (production stub, line 4485) |
|--------|-------------------------------|-------------------------------------------|
| `write_current_and_advance` | advance first, then write to new slot | write to current slot, then advance |
| `slot_handle` | wraps with `% self.capacity` | direct index (`slots[slot_index]`) |
| `new` invariant | `n_slots >= 1` | `slot_count >= 2` |

The test at line 4829 explicitly documents "advance-first semantics" confirming this is a deliberate divergence, not a mistake in one file. However, having two copies of the same type with incompatible contracts is a maintenance liability. Any future merge of `temp_edit.rs` into production creates a correctness trap: tests pass locally because they test their own version, but production callers expecting write-then-advance get advance-first behavior (or vice versa).

Severity is appropriately CRITICAL since this can cause silent history-slot corruption during the inevitable unification.

---

## HIGH

### H-01: `resources_freed` and `bytes_saved` double-count shared resources across eliminated passes

**VERDICT: REAL**

Verified at lines 3056-3066. The loop iterates over `eliminated_indices` (per eliminated pass) then per-pass writes. Two eliminated passes that both write the same resource `R` will each increment `resources_freed` and `bytes_saved` for `R`, even though only one unique resource is reclaimed. The `exclusively_used` check only verifies that no *live* pass reads the resource -- it does not deduplicate across eliminated passes.

The JUNIOR's suggested fix (use `HashSet<ResourceHandle>` to deduplicate) is correct. The severity is HIGH because `bytes_saved` feeds into allocator pressure heuristics.

### H-02: `write_current_and_advance` name does not match implementation in mod.rs

**VERDICT: OVERZEALOUS -- duplicate of C-01**

This is the same root cause presented as C-01 (the semantic inversion between the two files). The naming mismatch in mod.rs is a *consequence* of that inversion, not a separate finding. All actionable remediation (unify semantics, document contract, pick one direction) is already captured by C-01. Rolling the naming concern into C-01 as a secondary note is sufficient.

### H-03: `allocate_resources` silently drops unknown `ResourceDesc` variants

**VERDICT: REAL**

Verified at lines 4649-4736. The match handles `Texture2D`, `Texture3D`, and `Buffer`, then `_ => {}`. `ResourceDesc::TextureCube` is a valid variant of the enum (used in `texture_format` at line 2237 and `estimated_bytes` at line 834), but it is silently dropped by this match. Any `TextureCube` resource flowing through `allocate_resources` produces no allocation -- a downstream pass that depends on it gets an uninitialized handle.

The severity of HIGH is correct. This is a real correctness bug that manifests when TextureCube resources enter the allocator.

### H-04: `InterferenceGraph::build` ignores resources not in `lifetimes`, creating incorrect aliasing

**VERDICT: REAL**

Verified. `InterferenceGraph::build` (lines 2260-2310) only includes resources present in the `lifetimes` map. `greedy_color_resources` (lines 4873-4898) processes ALL resources in the input slice. Resources not in `lifetimes` get zero neighbors, so they receive color 0. In `allocate_resources`, since `greedy_color_resources` assigns a color to every resource, the `colors.get()` branch (not the else branch) activates -- the unlifetimed resource aliases with any other color-0 resource, even if their lifetimes overlap.

The existing test at line 7474 (`test_allocate_resource_handle_not_in_lifetimes`) tests a single resource with empty lifetimes, which masks the aliasing bug -- one resource aliased to itself is harmless. With two such resources with overlapping lifetimes, the aliasing is wrong and produces GPU read-after-write hazards. Severity HIGH is correct.

---

## MEDIUM

### M-01: `greedy_color_resources` assigns colors to imported and history resources unnecessarily

**VERDICT: OVERZEALOUS**

The performance concern is valid in theory but negligible in practice. Resources without lifetime entries (including imported resources) already get filtered out of the interference graph. The "wasted" work per resource is one empty `neighbors()` lookup (HashMap get) plus one insert of color 0 -- O(N) with trivial constants. For even the largest frame graphs (hundreds of resources), this costs microseconds per frame.

Removing this "waste" would require threading a filter through `greedy_color_resources` that distinguishes resource lifetime categories, adding code complexity for no measurable performance gain. Also, history resources with lifetime entries DO need coloring -- they alias with non-overlapping transient resources correctly. The finding incorrectly bundles history resources with imported ones.

### M-02: CullStats redundancy: `passes_eliminated` and `culled_pass_count` are always equal

**VERDICT: REAL**

Verified at lines 3077 and 3081: both are initialized from `eliminated.len()` at the single construction site. The struct field comment at line 2717 says "alias for passes_eliminated" confirming the redundancy is intentional but unnecessary. It wastes serialized JSON space and creates a maintenance liability where future changes must remember to keep both in sync. MEDIUM is appropriate.

### M-03: `else` branches in `allocate_resources` are dead code

**VERDICT: REAL**

Since `greedy_color_resources` assigns a color to every resource in the input slice (line 4895: `colors.insert(res.handle, color)`), `colors.get()` in the match arms at lines 4660, 4696, 4725 always returns `Some`. The `else` branches at lines 4674, 4710, 4730 are unreachable with the current call chain.

The JUNIOR correctly notes that these branches serve as an untested safety net: if someone refactors `greedy_color_resources` to skip certain resources, the fallback path silently produces standalone allocations that defeat aliasing. This is a real concern, not just dead code cleanup. MEDIUM severity is appropriate -- worth documenting but not urgent.

### M-04: CullStats Display test uses value assertions that match multiple fields

**VERDICT: REAL**

Verified at lines 10918-10934. `s.contains("3")` matches both `passes_eliminated=3` and `culled_pass_count=3` (and any other field serializing to "3"). No assertions check for field labels (`passes_total=`, `eliminated=`, etc.). The test would pass if one of the two `3` fields were removed or renamed to a different value, as long as the other "3" remained in the output. MEDIUM severity is appropriate for a test quality concern.

### M-05: `advance()` has `#[allow(dead_code)]` annotation inside test module

**VERDICT: REAL**

`advance()` is called at lines 7854, 7857, and 7860 within `mod tests`, so the Rust compiler would not emit a dead-code warning for it. The `#[allow(dead_code)]` annotation at line 4855 is stale. It likely survived from when the type was moved into the test module from production code. The annotation creates confusion about whether `advance()` is intentionally unused. MEDIUM severity is reasonable for a code hygiene issue.

---

## LOW

### L-01: Test comment misstates fallback when resource is not in lifetimes

**VERDICT: REAL**

The comment at line 7475 reads: "A transient resource without a lifetime entry falls back to (PassIndex(0), PassIndex(0)) -- still allocates correctly." But the `lifetimes` parameter is an empty `HashMap` and there is no fallback mechanism to `(PassIndex(0), PassIndex(0))`. The resource "works" because `InterferenceGraph::build` produces no edges for an empty lifetimes map, `greedy_color_resources` gives it color 0, and the color path in `allocate_resources` creates a standalone aliasing entry. The comment describes a nonexistent mechanism. LOW severity is correct.

### L-02: `num_colors` returns 0 for empty map instead of proper zero color count

**VERDICT: OVERZEALOUS**

The implementation `max().map(|m| m + 1).unwrap_or(0)` at line 4901 correctly returns the number of colors for all inputs producible by `greedy_color_resources`. The concern about "colors starting from color 1" is a theoretical edge case that cannot occur because `greedy_color_resources` always assigns color 0 as its starting point (line 4891: `let mut color = 0`). The suggested `.saturating_add(1)` is functionally identical for non-negative inputs. No practical impact.

### L-03: `PhysicalTexture` partial_eq excludes `handle` field, allowing accidental dedup of imported resources

**VERDICT: REAL**

`PartialEq` for `PhysicalTexture` (lines 4528-4535) compares format, width, height, depth, and is_transient but NOT the handle. In `AllocationTable::from_allocator` (line 4774), textures are deduplicated by equality. Two imported textures with matching dimensions/formats but different handles would collapse to a single physical index. This is currently latent because imported resources do not flow through `AllocationTable` in production, but any future production use of `from_allocator` with imported resources will silently alias them. LOW severity is appropriate given the latent nature.

### L-04: Missing tests for allocator edge cases

**VERDICT: REAL**

All six listed scenarios represent genuine coverage gaps:

| Scenario | REAL | Rationale |
|----------|------|-----------|
| `allocate_resources` with all-imported (no transient) | REAL | Verifies imported path works in isolation |
| Mixed Texture2D + Texture3D + Buffer aliasing | REAL | Cross-type aliasing correctness |
| Multiple resources of different sizes aliased to same color | REAL | Validates `or_insert_with` returns first-created physical resource |
| `AllocationTable::from_allocator` with imported-only | REAL | Imported dedup behavior (pairs with L-03) |
| Empty `lifetimes` with multiple transient resources | REAL | Exposes H-04 aliasing bug |
| 1-slot HistoryRingBuffer wrap-around | REAL | Degenerate case of `% 1` always returning 0 |

LOW severity is correct -- these are coverage gaps, not correctness bugs.

---

## Final Tally

| Verdict | Count | Findings |
|---------|-------|----------|
| **REAL** | 11 | C-01, H-01, H-03, H-04, M-02, M-03, M-04, M-05, L-01, L-03, L-04 |
| **OVERZEALOUS** | 3 | H-02 (duplicate of C-01), M-01 (negligible impact), L-02 (functionally correct) |

### Key Action Items

1. **C-01/H-02**: Unify the two `HistoryRingBuffer` implementations. Pick one semantic direction (write-then-advance is conventional), align both copies, and drop the stale `#[allow(dead_code)]` (M-05).

2. **H-01**: Deduplicate `resources_freed`/`bytes_saved` counting with a `HashSet<ResourceHandle>`.

3. **H-03**: Add a `TextureCube` arm to `allocate_resources` or replace `_ => {}` with an exhaustive match (possibly with `todo!()` for unhandled variants).

4. **H-04**: Either handle resources missing from `lifetimes` explicitly (warn/error), or ensure the test suite covers the multi-resource case to document the current behavior as intentional.

5. **M-02**: Remove `culled_pass_count` (deprecate in favor of `passes_eliminated`) or add a defensive assertion that they match if both are kept.

6. **L-03**: Include `handle` in `PhysicalTexture::eq` or add a distinct `compatible_with()` method for dedup logic that intentionally ignores handle.
