# Phase 2: Dependency DAG -- Architecture

## Overview
Builds the dependency graph between passes and produces a topological execution order. Also responsible for cycle detection. Covers S1-G4 (Dependency Analysis).

## Algorithm: DAG Builder (`build_dag`, mod.rs lines 1128-1204)

### Input
- `passes: &[IrPass]` -- all passes with their `ResourceAccessSet` (reads/writes)
- `_resources: &[IrResource]` (unused -- access sets are self-contained)

### Algorithm
1. For each pass, collect its resource accesses split into three categories:
   - ReadWrite resources (intersection of reads and writes)
   - Read-only resources (reads \ writes)
   - Write-only resources (writes \ reads)
2. For each resource, collect all `(pass_index, writes, reads)` tuples in insertion order
3. For every ordered pair `(i, j)` with `i < j` on the same resource:
   - Write(i) + Read(j) -> **RAW** edge (true data dependency)
   - Read(i) + Write(j) -> **WAR** edge (anti-dependency)
   - Write(i) + Write(j) -> **WAW** edge (output dependency)
4. Deduplicate via `HashSet<(usize, usize, ResourceHandle, EdgeType)>`

### Edge Types
| Type | Meaning | Example |
|------|---------|---------|
| RAW | Read-After-Write: B reads what A wrote | Shadow map -> Lighting |
| WAR | Write-After-Read: B overwrites what A read | Depth prepass -> Depth test |
| WAW | Write-After-Write: B overwrites what A wrote | Two post-processing passes |

## Algorithm: Topological Sort (`topological_sort`, mod.rs lines 1221-1305)

### Implementation: Kahn's Algorithm
1. Build adjacency list and in-degree map from edge list
2. Seed BFS queue with all zero-in-degree passes
3. While queue not empty: pop pass, add to order, decrement in-degree of successors, enqueue any newly zero-in-degree
4. If processed count < total passes -> **cycle detected** (returns error string)
5. Tie-breaking: FIFO order (stable insertion-order processing)

### Output
- `Result<Vec<PassIndex>, String>` -- ordered pass indices or cycle error

## Python Implementation

Python `_build_dependency_graph()` (frame_graph.py lines 497-519) is simplified:
- Iterates passes in declaration order
- For each resource access, finds the producer pass
- Creates `{consumer: [producers]}` dependency mapping
- No Kahn's algorithm; no topological sort; uses declaration order after culling

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-2.1 | [x] | mod.rs:1128-1204 |
| T-FG-2.2 | [x] | mod.rs:1221-1305 |
| T-FG-2.3 | [x] | mod.rs:1299-1305 (cycle detection) |
| T-FG-2.4 | [-] | Not implemented |
| T-FG-2.5 | [-] | Not implemented |
| T-FG-2.6 | [x] | mod.rs test module (4 DAG + 4 topo tests) |
| T-FG-2.7 | [~] | No 10-pass/20-edge benchmark |

## Gaps & Risks

1. **No topological depth** -- Longest-path-from-entry is not computed, needed for parallel region identification and scheduling
2. **No parallel regions** -- Passes at same depth with no path between them cannot be identified; needed for GPU parallel execution planning
3. **Cycle diagnostics are generic** -- "Cycle detected" error gives no resource-level path information
4. **Python topological sort is simplified** -- Uses declaration order, not Kahn's algorithm; this is incorrect when passes are registered out-of-order
5. **Tests missing edge coverage** -- No diamond DAG, no WAW-only edge test, no single-pass DAG test
