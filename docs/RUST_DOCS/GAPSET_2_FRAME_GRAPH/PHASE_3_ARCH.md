# Phase 3: Resource Aliasing -- Architecture

## Overview
Computes resource lifetimes and enables memory sharing (aliasing) between transient resources with non-overlapping lifetimes. Also manages history and external resources. Covers S1-G5 (Resource Aliasing).

## Algorithm: Lifetime Computation (`compute_lifetimes`, mod.rs lines 1310-1406)

### Input
- `passes: &[IrPass]` -- all passes in topological order
- `edges: &[IrEdge]` -- dependency edges (unused in current implementation)
- `resources: &[IrResource]` -- all resources

### Output: `Vec<ResourceLifetime>` where ResourceLifetime contains:
- `first_pass: PassIndex` -- index of the first pass that reads or writes this resource
- `last_pass: PassIndex` -- index of the last pass that reads or writes this resource
- `is_read: bool` -- resource is read at least once
- `is_written: bool` -- resource is written at least once

### Algorithm
1. Initialize all resources with `(MAX, MIN, false, false)`
2. For each pass in order:
   - For each resource in pass reads: update first_pass (min), last_pass (max), is_read
   - For each resource in pass writes: update first_pass (min), last_pass (max), is_written
3. Include color attachment resources (load_op=Load -> read, store_op=Store -> write)
4. Include depth/stencil resources (depth_load_op, stencil_load_op -> read; depth_store_op, stencil_store_op -> write)

## Algorithm: Memory Aliasing (Python `compute_aliasing`, resource_manager.py lines 563-609)

### Algorithm
1. Sort transients by `(first_use_pass, last_use_pass)`
2. For each transient:
   - Check against existing alias groups
   - If all members have non-overlapping lifetimes (`overlaps_with()` returns false), add to group
   - If no compatible group, create a new group
3. Resources with overlapping lifetimes cannot alias

### Aliasing Rule
Two resources can alias iff their lifetimes do NOT overlap:
- `last_use_pass < other.first_use_pass` OR `other.last_use_pass < first_use_pass`
- Format and dimension compatibility is NOT checked

## Three Resource Types

### Transient Resources
- Allocated per-frame, eligible for aliasing
- Managed by `TransientResource` (handle, first_use, last_use, alias_group, state)

### History Resources
- Persisted across frames for temporal effects (TAA, GI accumulators)
- `HistoryResource` with `double_buffered: bool` and `swap_buffers()`
- Frame N uses index `N % 2` -- not a generalized N-slot ring buffer

### External Resources
- Imported from outside (swap chain backbuffer, asset textures)
- `ExternalResource` with opaque `gpu_resource` handle
- No allocation -- state tracking only

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-3.1 | [x] | mod.rs:1310-1406 |
| T-FG-3.2 | [~] | resource_manager.py:563-609 (no explicit interference graph) |
| T-FG-3.3 | [~] | resource_manager.py:578-609 (interval heuristic, not graph coloring) |
| T-FG-3.4 | [-] | Not implemented |
| T-FG-3.5 | [~] | resource_manager.py:246-284 (double-buffer, not ring buffer) |
| T-FG-3.6 | [x] | resource_manager.py:436-485 |
| T-FG-3.7 | [-] | Not implemented |
| T-FG-3.8 | [~] | 1 Rust test, 6 Python resource tests |
| T-FG-3.9 | [-] | Not implemented |

## Gaps & Risks

1. **No wgpu resource allocation** -- The `ResourceAllocator` does not exist. No actual GPU textures or buffers are created from alias groups
2. **No allocation table** -- No mapping from `ResourceHandle` to `(wgpu::Texture, u32 layer_or_offset)`
3. **No format/dimension compatibility** -- Current aliasing only checks lifetime overlap, not format compatibility or dimension compatibility
4. **Graph coloring is a heuristic** -- Python's interval heuristic is not the specified greedy largest-first coloring; may give suboptimal memory savings
5. **History ring buffer is limited** -- Only 2-slot double-buffering, not the N-slot generalized ring buffer
6. **Memory savings cannot be measured** -- Without wgpu allocation, the 40%+ savings target (T-FG-3.9) cannot be verified
