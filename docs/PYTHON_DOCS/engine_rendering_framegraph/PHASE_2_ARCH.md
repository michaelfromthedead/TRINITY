# PHASE 2 ARCHITECTURE: Resource Allocation Integration

## Overview

Connect the resource aliasing system to actual GPU memory allocation. The `ResourceManager` already computes alias groups and lifetimes; this phase makes `allocate_transient()` real.

## Architecture Decisions

### ADR-FG-005: Heap-Per-Alias-Group Strategy

**Decision**: Each alias group gets one GPU heap allocation. Members share the heap via offset/size.

**Rationale**:
- Alias groups are already computed by `compute_aliasing()`
- One heap per group minimizes allocation calls
- Members reference heap + offset, not individual allocations

**Consequences**:
- Heap size = max(member sizes) in group
- All members in group must be compatible (same memory type)
- Fragmentation within groups is zero (sequential use)

### ADR-FG-006: Deferred Allocation

**Decision**: `allocate_transient()` returns handle immediately; actual GPU allocation happens at `begin_frame()`.

**Rationale**:
- Frame graph compilation happens before execution
- Aliasing computation requires all resources declared first
- Backend can batch allocations for efficiency

**Consequences**:
- Handles are valid but "pending" until frame start
- Out-of-memory errors surface at `begin_frame()`, not declaration time
- Backend must track pending vs. allocated state

### ADR-FG-007: Memory Type Inference

**Decision**: Infer GPU memory type from resource usage flags.

| Usage | Memory Type |
|-------|-------------|
| RENDER_TARGET, DEPTH_STENCIL | Device-local (GPU heap) |
| UPLOAD | Host-visible, write-combined |
| READBACK | Host-visible, cached |
| TRANSIENT | Device-local, may alias |

**Rationale**:
- Explicit memory type selection is error-prone
- Usage flags already declared per resource
- Backend maps to Vulkan/D3D12 memory types

**Consequences**:
- Advanced users can't override memory type directly
- Future: add `memory_hint` field if needed
- Validation layer can warn on incompatible usage combos

### ADR-FG-008: History Resource Double-Buffering

**Decision**: `HistoryResource` allocates two physical buffers; `begin_frame()` swaps indices.

**Rationale**:
- History resources persist across frames (TAA, motion vectors)
- Double-buffering prevents read/write hazards
- Swap is index flip, not data copy

**Consequences**:
- History resources use 2x memory of declared size
- `get_current()` and `get_previous()` return correct buffer per frame
- Barrier manager must track both buffers' states

## Component Diagram

```
+---------------------+
|  ResourceManager    |
|  (alias groups)     |
+----------+----------+
           |
           | calls
           v
+----------+----------+
|  RHIContext         |
|  allocate_transient |
+----------+----------+
           |
           | backed by
           v
+----------+----------+
|  HeapAllocator      |  <-- per-alias-group heaps
|  (GPU memory)       |
+---------------------+
```

## Data Flow

1. **Declaration**: `ResourceManager.create_transient()` records descriptor
2. **Aliasing**: `ResourceManager.compute_aliasing()` groups by lifetime
3. **Allocation**: `RHIContext.allocate_transient()` called per group (not per resource)
4. **Binding**: `AllocationHandle` carries (heap_id, offset, size)
5. **Execution**: Pass callbacks receive bound resources

## Files Affected

- `engine/rendering/framegraph/resource_manager.py` — call context allocation methods
- `engine/rendering/framegraph/context.py` — refine `allocate_transient()` signature
- `engine/rendering/rhi/heap_allocator.py` — new file for GPU heap management
- `engine/rendering/rhi/wgpu_context.py` — implement allocation (Phase 3)
