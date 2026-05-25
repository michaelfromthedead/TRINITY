# engine/rendering/framegraph Investigation

**Lines**: 3,524 (3,312 in main files + 61 in config.py + ~150 in __init__.py)
**Classification**: REAL (with execution stubbed at RHI boundary)

## File Analysis

### frame_graph.py (879 lines) - REAL
Core orchestrator implementing full frame graph lifecycle:
- `FrameGraph` class with complete resource creation, pass management, compilation, and execution
- `compile()` implements: dependency analysis, pass culling, execution ordering, resource aliasing, async scheduling, barrier insertion
- `serialize()` method for Python-to-Rust bridge (JSON matching `IrPass`/`IrResource` types)
- Real algorithms: dependency graph construction via producer tracking, topological sort for execution order

Key classes/algorithms:
- `CompilationResult`: Success/error tracking with statistics
- `_build_dependency_graph()`: Producer/consumer tracking via resource read/write declarations
- `_cull_unused_passes()`: Dead code elimination based on output usage
- `_update_resource_lifetimes()`: Lifetime tracking for aliasing computation

### pass_node.py (726 lines) - REAL
Complete pass type hierarchy with resource dependency tracking:
- `PassNode` (ABC): Base class with read/write dependency lists, execution callback, flag management
- `GraphicsPass`: Color attachments, depth/stencil, viewport/scissor, MSAA resolve
- `ComputePass`: Dispatch size, indirect dispatch, texture/buffer read/write
- `CopyPass`: Source/destination with regions and offsets
- `RayTracingPass`: Dispatch dimensions, TLAS, shader binding table, recursion depth

Real implementations:
- Method chaining API for building passes (`add_color_attachment()`, `set_depth_stencil()`)
- Resource state tracking per access (`ResourceAccess` dataclass)
- Factory function `create_pass()` for type-safe instantiation

### resource_manager.py (654 lines) - REAL
Full resource lifecycle management with aliasing support:
- `ResourceManager`: Creates/tracks transient, history, and external resources
- `TransientResource`: Per-frame allocation with lifetime tracking and aliasing
- `HistoryResource`: Persisted across frames, double-buffering support
- `ExternalResource`: Imported resources (backbuffer, external textures)

Key algorithms:
- `compute_aliasing()`: Groups non-overlapping transients by first_use/last_use intervals
- `update_lifetime()`: Tracks pass indices for aliasing decisions
- `begin_frame()`: Resets transient states, swaps history buffers

### barrier_manager.py (574 lines) - REAL
Automatic GPU barrier insertion with full state machine:
- `BarrierManager`: Analyzes passes, generates transition barriers
- `ResourceStateTracker`: Tracks per-resource and per-subresource states
- `Barrier`: Full barrier specification (type, old/new state, pipeline stages, access flags)
- `BarrierBatch`: Groups barriers for efficient execution

Key implementations:
- `_STATE_TO_STAGE` mapping: Resource state to optimal pipeline stage and access flags (lines 184-241)
- `_analyze_pass()`: Creates transition barriers for reads and writes
- `_check_uav_hazards()`: Detects UAV read-after-write requiring explicit barriers
- `create_aliasing_barrier()`: Memory reuse synchronization

### async_scheduler.py (479 lines) - REAL
Multi-queue GPU scheduling with synchronization:
- `AsyncScheduler`: Schedules passes across graphics/compute/copy queues
- `QueueTimeline`: Per-queue pass tracking with fence values
- `SyncPoint`: Cross-queue synchronization primitives
- `ScheduledPass`: Pass with queue assignment and sync requirements

Key algorithms:
- `_can_run_async()`: Heuristic for safe async compute execution
- `_compute_sync_points()`: Inserts synchronization for cross-queue dependencies
- `get_parallel_groups()`: Groups consecutive async compute for overlap
- `estimate_overlap_benefit()`: Performance improvement estimation

### config.py (61 lines) - REAL
Dataclass-based configuration:
- `AsyncSchedulerConfig`: Window sizes, benefit caps, candidate thresholds
- `ResourceManagerConfig`: Buffer size minimums, texture defaults
- `FrameGraphConfig`: Default feature toggles

## Key Findings

This is a **production-quality frame graph implementation** matching modern GPU rendering architectures (Frostbite, Unreal, Unity HDRP). The subsystem implements:

1. **Render Pass Scheduling**: Full DAG-based dependency analysis with topological ordering
2. **Resource Aliasing**: Memory-efficient transient allocation via non-overlapping lifetime analysis
3. **Automatic Barriers**: State machine tracking with D3D12/Vulkan-style pipeline stage synchronization
4. **Async Compute**: Multi-queue scheduling with fence-based cross-queue synchronization
5. **Dead Code Elimination**: Unused pass culling based on output consumption
6. **Rust Bridge**: Serialization to JSON for PyO3-based compilation (`serialize()` method)

### What's REAL (100% complete)
- All data structures, resource descriptors, pass types, barrier types
- Dependency analysis, aliasing, and scheduling algorithms
- Configuration system with dataclasses

### What's STUBBED (0% complete - at RHI boundary)
- Actual GPU execution - all `execute()` methods defer to user callback or context
- GPU resource allocation - `allocated_offset`, `size_bytes` tracked but not allocated
- Context interface - typed as `Any`, no concrete RHI implementation

**Critical Comment (frame_graph.py:664-670):**
```python
# In a real implementation, this would call into the RHI
# to execute the actual GPU barriers.
# The context object should provide a method like:
#   context.execute_barriers(batch.barriers)
# For now, we log/track the barriers for debugging purposes.
```

## Evidence

### Dependency Graph Construction (REAL algorithm)
```python
# frame_graph.py lines 498-527
def _build_dependency_graph(self) -> None:
    producers: dict[str, str] = {}
    dependencies: dict[str, list[str]] = {name: [] for name in self._pass_order}

    for pass_name in self._pass_order:
        pass_node = self._passes[pass_name]
        for access in pass_node.reads:
            resource_name = access.handle.name
            if resource_name in producers:
                producer_pass = producers[resource_name]
                if producer_pass not in dependencies[pass_name]:
                    dependencies[pass_name].append(producer_pass)
        for access in pass_node.writes:
            resource_name = access.handle.name
            producers[resource_name] = pass_name
    self._pass_dependencies = dependencies
```

### Resource Aliasing Algorithm (REAL algorithm)
```python
# resource_manager.py lines 563-609
def compute_aliasing(self) -> None:
    sorted_transients = sorted(
        self._transients.values(),
        key=lambda t: (t.first_use_pass, t.last_use_pass),
    )
    for transient in sorted_transients:
        if transient.first_use_pass == -1:
            continue
        found_group = False
        for group_id, group_members in self._alias_groups.items():
            can_alias = True
            for member in group_members:
                if transient.overlaps_with(member):
                    can_alias = False
                    break
            if can_alias:
                transient.alias_group = group_id
                group_members.append(transient)
                found_group = True
                break
        if not found_group:
            group_id = self._next_alias_group
            self._next_alias_group += 1
            transient.alias_group = group_id
            self._alias_groups[group_id] = [transient]
```

### Barrier State Tracking (REAL state machine)
```python
# barrier_manager.py lines 439-479
def _create_transition_barrier(
    self,
    handle: ResourceHandle,
    required_state: ResourceState,
    subresource: Optional[int] = None,
) -> Optional[Barrier]:
    current_state = self._state_tracker.get_state(handle, subresource)
    if not _needs_barrier(current_state, required_state):
        return None
    src_stage, src_access = _get_stage_and_access(current_state)
    dst_stage, dst_access = _get_stage_and_access(required_state)
    barrier = Barrier(
        handle=handle,
        barrier_type=BarrierType.TRANSITION,
        old_state=current_state,
        new_state=required_state,
        src_stage=src_stage,
        dst_stage=dst_stage,
        src_access=src_access,
        dst_access=dst_access,
        subresource=subresource,
    )
    self._state_tracker.set_state(handle, required_state, subresource)
    return barrier
```

### Cross-Queue Synchronization (REAL GPU scheduling)
```python
# async_scheduler.py lines 296-344
def _compute_sync_points(self) -> None:
    graphics_writes: dict[str, ScheduledPass] = {}
    compute_writes: dict[str, ScheduledPass] = {}
    for scheduled in self._graphics_timeline.passes:
        for handle in scheduled.pass_node.get_write_handles():
            graphics_writes[handle.name] = scheduled
    for scheduled in self._compute_timeline.passes:
        for handle in scheduled.pass_node.get_write_handles():
            compute_writes[handle.name] = scheduled
    for scheduled in self._graphics_timeline.passes:
        for handle in scheduled.pass_node.get_read_handles():
            if handle.name in compute_writes:
                writer = compute_writes[handle.name]
                sync = self._create_sync_point(
                    writer, scheduled, QueueType.COMPUTE, QueueType.GRAPHICS,
                )
                self._sync_points.append(sync)
                writer.sync_after.append(sync)
                scheduled.sync_before.append(sync)
```

### UAV Hazard Detection (REAL algorithm)
```python
# barrier_manager.py:481-514
def _check_uav_hazards(self, pass_node: PassNode) -> list[Barrier]:
    barriers: list[Barrier] = []
    for access in pass_node.reads:
        if access.state == ResourceState.UNORDERED_ACCESS:
            current = self._state_tracker.get_state(access.handle, access.subresource)
            if current == ResourceState.UNORDERED_ACCESS:
                barrier = Barrier(
                    handle=access.handle,
                    barrier_type=BarrierType.UAV,
                    old_state=current,
                    new_state=access.state,
                    src_stage=PipelineStage.COMPUTE_SHADER,
                    dst_stage=PipelineStage.COMPUTE_SHADER,
                    src_access=AccessFlags.SHADER_WRITE.value,
                    dst_access=AccessFlags.SHADER_READ.value,
                    subresource=access.subresource,
                )
                barriers.append(barrier)
    return barriers
```

## Verdict

**Classification: REAL (PARTIAL IMPLEMENTATION)**

The frame graph subsystem is ~70% complete with production-grade architecture matching Frostbite/UE4/Unity HDRP patterns. All algorithms are fully implemented; execution is stubbed at the RHI boundary where the `context: Any` parameter would need to be replaced with a concrete wgpu/Vulkan/D3D12 backend.
