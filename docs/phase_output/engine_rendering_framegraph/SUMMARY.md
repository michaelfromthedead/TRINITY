# engine/rendering/framegraph — Quantitative Summary

## Metrics

| Metric | Value |
|--------|-------|
| **Total Lines** | 3,524 |
| **Main Files Lines** | 3,312 |
| **Config Lines** | 61 |
| **Init Lines** | ~150 |
| **Files** | 6 |
| **Classes** | 15+ |
| **Key Functions** | 30+ |

## File Breakdown

| File | Lines | Classification |
|------|-------|----------------|
| frame_graph.py | 879 | REAL |
| pass_node.py | 726 | REAL |
| resource_manager.py | 654 | REAL |
| barrier_manager.py | 574 | REAL |
| async_scheduler.py | 479 | REAL |
| config.py | 61 | REAL |
| __init__.py | ~150 | REAL |

## Algorithm Inventory

| Algorithm | Location | Lines | Status | Complexity |
|-----------|----------|-------|--------|------------|
| Dependency Graph Construction | frame_graph.py:498-527 | 30 | REAL | O(P*R) |
| Topological Sort | frame_graph.py:compile() | ~50 | REAL | O(P+E) |
| Pass Culling | frame_graph.py:_cull_unused_passes() | ~40 | REAL | O(P) |
| Resource Lifetime Update | resource_manager.py:update_lifetime() | ~20 | REAL | O(1) |
| Resource Aliasing | resource_manager.py:563-609 | 47 | REAL | O(T*G) |
| Barrier State Tracking | barrier_manager.py:439-479 | 41 | REAL | O(1) |
| UAV Hazard Detection | barrier_manager.py:481-514 | 34 | REAL | O(A) |
| Aliasing Barrier Creation | barrier_manager.py:create_aliasing_barrier() | ~25 | REAL | O(1) |
| Cross-Queue Sync Computation | async_scheduler.py:296-344 | 49 | REAL | O(P*W) |
| Async Compute Heuristic | async_scheduler.py:_can_run_async() | ~30 | REAL | O(D) |
| Parallel Group Detection | async_scheduler.py:get_parallel_groups() | ~40 | REAL | O(P) |
| Overlap Benefit Estimation | async_scheduler.py:estimate_overlap_benefit() | ~25 | REAL | O(G) |

Legend: P=passes, R=resources, E=edges, T=transients, G=groups, A=accesses, W=writes, D=dependencies

## Class Inventory

| Class | File | Purpose |
|-------|------|---------|
| FrameGraph | frame_graph.py | Core orchestrator |
| CompilationResult | frame_graph.py | Compile success/error tracking |
| PassNode (ABC) | pass_node.py | Base pass class |
| GraphicsPass | pass_node.py | Rasterization pass |
| ComputePass | pass_node.py | Compute shader pass |
| CopyPass | pass_node.py | Memory transfer pass |
| RayTracingPass | pass_node.py | Ray tracing pass |
| ResourceAccess | pass_node.py | Per-access state tracking |
| ResourceManager | resource_manager.py | Resource lifecycle |
| TransientResource | resource_manager.py | Per-frame allocation |
| HistoryResource | resource_manager.py | Cross-frame persistence |
| ExternalResource | resource_manager.py | Imported resources |
| BarrierManager | barrier_manager.py | Barrier generation |
| ResourceStateTracker | barrier_manager.py | State machine |
| Barrier | barrier_manager.py | Barrier specification |
| BarrierBatch | barrier_manager.py | Grouped barriers |
| AsyncScheduler | async_scheduler.py | Multi-queue scheduler |
| QueueTimeline | async_scheduler.py | Per-queue tracking |
| SyncPoint | async_scheduler.py | Cross-queue sync |
| ScheduledPass | async_scheduler.py | Pass with queue assignment |

## Completion Assessment

| Component | Completion |
|-----------|------------|
| Data Structures | 100% |
| Algorithms | 100% |
| Configuration | 100% |
| Serialization | 100% |
| RHI Integration | 0% |
| **Overall** | ~70% |
