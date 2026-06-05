# Phase 6: Dead Pass Elimination -- Architecture

## Overview
Removes passes whose outputs are never consumed by any downstream pass. Optimizes GPU time by skipping unnecessary work. Covers S1-G8 (Dead Pass Elimination).

## Algorithm: Dead Pass Elimination (`eliminate_dead_passes`, mod.rs lines 1648-1722)

### Input
- `passes: Vec<IrPass>` -- all passes
- `order: &[PassIndex]` -- topological ordering
- `edges: &[IrEdge]` -- dependency edges (unused in current implementation)

### Algorithm: Reverse Reachability
1. Build `resource_readers: HashMap<ResourceHandle, Vec<PassIndex>>` -- for each resource, the set of passes that read it
2. For each pass:
   - Skip passes with empty write sets (no outputs to cull)
   - **Never eliminate Graphics passes** (conservative -- they always produce the frame)
   - For Compute/Copy passes: check if ALL write resources have no downstream readers (other than the pass itself)
   - If `all_unread && !writes.is_empty()` -> mark dead
3. Second pass: ensure no graphics pass is ever marked dead (belt-and-suspenders)
4. Build pruned order: filter out dead passes

### Output
- `(passes: Vec<IrPass>, pruned_order: Vec<PassIndex>, eliminated: Vec<PassIndex>)`

### Python Implementation (`_cull_unused_passes`, frame_graph.py lines 502-562)
- Uses backbuffer to determine the "live output" set (the external resource that must be produced)
- Culls a pass if:
  - No cull flag prevents it (`NO_CULL` or `SIDE_EFFECTS` flags)
  - None of its writes are in the transitive live set
- `_compute_transitive_dependencies()`: BFS from backbuffer through dependency chain
- Supports `_enable_pass_culling: bool` to disable entirely

### PassFlags for Culling Control (Python `pass_node.py`)
| Flag | Effect |
|------|--------|
| `NO_CULL` | Pass is never removed by dead pass elimination |
| `SIDE_EFFECTS` | Pass has external effects (dispatch indirect, UAV writes not tracked as resources) |

## Preservation Rules

| Pass Type | Always Live? | Reason |
|-----------|-------------|--------|
| Graphics | Yes (Rust) / Depends (Python) | Produces rendered output |
| With NO_CULL | Yes | Explicitly marked |
| With SIDE_EFFECTS | Yes | May have untracked outputs |
| Writes backbuffer | Yes (Python) | Final output to screen |
| History resource writer | Not checked | Should be always live per spec |

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-6.1 | [~] | mod.rs:1678-1681 (always-live graphics); Python:598-562 (backbuffer-based) |
| T-FG-6.2 | [x] | mod.rs:1653-1693 (resource_readers map) |
| T-FG-6.3 | [x] | mod.rs:1703-1721 (pruned order) |
| T-FG-6.4 | [~] | Python NO_CULL/SIDE_EFFECTS flags; no runtime FeatureSet |
| T-FG-6.5 | [x] | mod.rs:2467-2475, 2484, 2688-2690, 3857-3876 (CullStats struct, Display, JSON emit, populate) |
| T-FG-6.6 | [~] | 5 Python tests; 0 Rust tests |
| T-FG-6.7 | [-] | Not implemented |

## Gaps & Risks

1. **No explicit live output set** -- Rust uses an implicit "graphics passes are always live" rule. The spec requires an explicit `Vec<ResourceHandle>` for swap chain, history, and debug outputs
2. **No dynamic culling** -- Python supports `NO_CULL`/`SIDE_EFFECTS` flags but no runtime `FeatureSet` bitfield for debug pass toggling
3. **Rust has zero dead pass elimination tests** -- All testing is in Python
4. **No resource reclamation** -- Resources exclusively used by dead passes are not removed from the allocation table (which doesn't exist yet)
5. **Conservative graphics rule may be too conservative** -- A graphics pass writing only to an unused intermediate target could theoretically be culled, but the current implementation never attempts this
