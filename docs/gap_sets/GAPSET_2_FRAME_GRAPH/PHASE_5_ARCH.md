# Phase 5: Async Compute Scheduling -- Architecture

## Overview
Identifies compute passes that can run on a secondary compute queue in parallel with the main graphics timeline. Covers S1-G7 (Async Scheduling).

## Key Types

### QueueType (Python `async_scheduler.py:26`)
| Variant | Purpose |
|---------|---------|
| `GRAPHICS` | Main queue -- supports all operations |
| `COMPUTE` | Async compute queue -- compute dispatches only |
| `COPY` | Transfer/copy queue -- copy operations only |

### SyncPoint (Python `async_scheduler.py:40`)
| Field | Description |
|-------|-------------|
| `name` | Human-readable name |
| `signal_queue` | Queue that signals completion |
| `wait_queue` | Queue that waits for completion |
| `signal_pass` | Pass that signals (the async compute pass) |
| `wait_pass` | Pass that waits (the dependent graphics pass) |
| `fence_value` | Timeline fence value for GPU synchronization |

### ScheduledPass (Python `async_scheduler.py:67`)
| Field | Description |
|-------|-------------|
| `pass_node` | The underlying PassNode |
| `queue` | Assigned QueueType |
| `sync_before` | SyncPoints to wait for before execution |
| `sync_after` | SyncPoints to signal after execution |
| `parallel_group` | Group ID for parallel execution (-1 = sequential) |

## Algorithm: Async Eligibility (`async_schedule`, mod.rs lines 1576-1632)

### Rust Implementation
1. Build `raw_writers: Vec<Vec<PassIndex>>` -- for each pass, which other passes write to resources it reads (RAW edges only)
2. Iterate passes in topological order:
   - Skip non-Compute and non-Copy passes
   - For each incoming RAW edge, check if the writer is a Graphics or RayTracing pass
   - If any writer is Graphics/RayTracing -> blocked (must stay on main queue)
   - If no blockers -> eligible for async queue

### Python Implementation (`async_scheduler.py:100-281`)
- `_determine_queue()`: Copy->COPY, Compute->COMPUTE if `ASYNC_COMPUTE` flag or `_can_run_async()`, Graphics/RT->GRAPHICS
- `_can_run_async()`: check recent graphics writes within configurable window (default 3 passes)
- `identify_async_candidates()`: standalone heuristic function
- `_compute_sync_points()`: creates SyncPoint entries for each resource written by async compute and read by downstream graphics

## Feature Gating (Not Implemented)

The T-FG-5.4 spec requires checking `wgpu::Features::TIMELINE_SEMAPHORE` at compile time. Current implementation uses a Python `enable_async_compute: bool` config flag with no wgpu feature check.

## Serial Fallback (Not Implemented)

When async compute is unavailable, eligible compute passes should be flattened onto the graphics timeline. Current implementation simply skips async scheduling entirely when disabled.

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-5.1 | [x] | Rust mod.rs:1576-1632, Python async_scheduler.py:208-233 |
| T-FG-5.2 | [~] | Rust flat list, Python QueueTimeline grouping |
| T-FG-5.3 | [~] | Python SyncPoint computation exists; not wired to wgpu |
| T-FG-5.4 | [-] | Not implemented |
| T-FG-5.5 | [-] | Not implemented |
| T-FG-5.6 | [~] | Python: full types; Rust: enum + tuples |
| T-FG-5.7 | [-] | Not implemented |
| T-FG-5.8 | [-] | Not implemented |

## Gaps & Risks

1. **No feature gating** -- Cannot detect platform support for async compute. The `enable_async_compute` flag is a best-effort toggle with no wgpu capability check
2. **No serial fallback** -- Disabling async compute skips scheduling entirely rather than gracefully flattening passes. This means disabled async compute changes the execution order
3. **Sync points not wired** -- Python computes SyncPoint entries but never translates them to wgpu semaphore operations. Cross-queue synchronization currently exists only on paper
4. **No async tests** -- Zero unit tests in either codebase for async scheduling
5. **Rust async_schedule uses pattern matching on pass_type** -- Uses `PassType::Compute { .. }` which fails to match because `PassType` is a simple enum, not a struct variant. This may produce incorrect results
6. **Window-based heuristic is fragile** -- Python's 3-pass window for recent writes may miss dependencies in frames with many compute passes
