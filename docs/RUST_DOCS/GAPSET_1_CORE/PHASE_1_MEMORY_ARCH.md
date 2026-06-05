# PHASE 1: Memory Management + Entity System

**Scope:** Implement frame-scoped bump allocation, pooled allocation, LIFO stack allocation, and the generational-index entity identifier.
**Depends on:** Phase 0 (omega crate for base types, bytemuck dependency)
**Produces:** Three allocator strategies in `renderer-backend/src/memory.rs`; Python `Entity` + `EntityAllocator` in `engine/core/ecs/entity.py`
**Status:** MOSTLY COMPLETE (3/5 tasks DONE, 2 PARTIAL)

## 1. Overview

Phase 1 provides the memory management primitives and entity identification system for the TRINITY engine core. Three allocator strategies serve distinct GPU memory allocation patterns:

- **FrameAllocator (bump-pointer):** Transient per-frame allocations reset at frame boundaries. No individual deallocation -- the entire buffer is reclaimed by `reset()`. Ideal for per-frame staging data, command buffer scratch, and temporary GPU writes.
- **PoolAllocator (block-size classes):** Fixed-size pools at 64KB, 256KB, 1MB, 4MB. `acquire()`/`release()` via free-list tracking. For predictable-size allocations that persist beyond a single frame.
- **StackAllocator (LIFO):** Nested allocation with LIFO semantics. For hierarchical staging operations where lifetimes form a proper stack.

Entity identification uses a generational index: 24-bit index + 16-bit generation packed into a single int, implemented in Python as `Entity` + `EntityAllocator` with generation bumping on deallocation.

The GPU budget tracker (`GpuBudget`) provides atomic capacity planning: total, used, and available bytes tracked atomically for cross-thread visibility.

## 2. Architectural decisions

- **FrameAllocator (bump) as primary transient allocator.** Matches the frame-bound lifetime of render data. Bump-pointer is O(1) allocate, O(1) reset, with zero fragmentation. Divergence from original plan: named `FrameAllocator` instead of `LinearAllocator`; same semantics.
- **PoolAllocator uses block-size classes, not uniform slots.** The original spec described fixed-size uniform slots; the actual implementation uses size-class pools (64KB/256KB/1MB/4MB). This serves the same purpose (O(1) acquire/release with predictable allocation) while accommodating variable-size GPU resource needs.
- **StackAllocator (LIFO) replaces RingBuffer.** The original spec described a circular ring buffer with head/tail cursors. The actual `StackAllocator` uses LIFO semantics, which is simpler and sufficient for nested staging. The triple-buffered `BufferRegistry` in `gpu_driven/buffers.rs` covers the GPU staging circular-buffer use case.
- **EntityId is Python-only for now.** The Rust bridge uses `entity_id: u64` (plain integer, no generation bits). The Python `Entity` class packs index (24 bits) + generation (16 bits) into a single int, but no Rust-side generation checking exists. This means stale-entity-id detection is Python-only.
- **All allocators in a single file.** `memory.rs` (rendered in renderer-backend) contains all three allocators plus the GPU budget tracker. Total ~15 inline tests.

## 3. Constraints specific to this phase

- All allocations must be CACHE_LINE (64-byte) aligned to prevent false sharing across threads.
- FrameAllocator reset must make all memory available for the next frame without individual deallocation.
- PoolAllocator free-list must be O(1) acquire/release.
- Entity generation bits must wrap at 65535 to prevent runaway.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `crates/renderer-backend/src/memory.rs` | FrameAllocator (bump, per-frame reset), PoolAllocator (64KB/256KB/1MB/4MB block classes), StackAllocator (LIFO), GpuBudget (atomic capacity tracking). 15 inline tests. | DONE (FrameAllocator, PoolAllocator, StackAllocator) |
| `engine/core/ecs/entity.py` | `Entity` (index+generation packed into int), `EntityAllocator` (allocate/deallocate with generation bump, free-list reuse, null sentinel). | DONE |
| `omega/src/bridge.rs` | Uses `entity_id: u64` (plain integer, no generation checking). | PARTIAL -- No generational index type in Rust |
| `crates/renderer-backend/src/gpu_driven/buffers.rs` | `BufferRegistry` triple-buffered staging (Idle/Acquired/Submitted/Ready). | DONE (covers GPU staging need originally assigned to RingBuffer) |

**Notable divergence -- RingBuffer/StackAllocator:**
The original spec described a circular `RingBuffer` with head/tail cursors for staging. The actual implementation uses `StackAllocator` (LIFO for nested staging) and `BufferRegistry` (triple-buffered for GPU staging). The staging use case is fully covered, but not via the mechanism originally specified.

**Notable divergence -- EntityId generation:**
No `bytemuck::Pod + Zeroable` Rust `EntityId` type with 24-bit index + 8-bit generation exists. The bridge passes `entity_id: u64` as a plain integer. The Python `Entity` has generation tracking, but stale-ID detection is not possible from Rust. Future Rust ECS work should implement a proper generational index.

## 5. Testing strategy

- 15 inline tests in `memory.rs` covering: FrameAllocator allocation/alignment/reset/high-water-mark, PoolAllocator acquire/release/free-list-reuse, StackAllocator push/pop LIFO ordering.
- Python entity tests in the ECS test suite cover: entity allocation, deallocation with generation bump, null entity handling, MAX_ENTITIES overflow.
- No Rust EntityId tests exist (the type doesn't exist in Rust).

## 6. Open questions

- Should a proper Rust `EntityId` (u32 with 24-bit index + 8-bit generation, Pod/Zeroable) be added to omega? This would enable generation checking on the Rust side and allow bytemuck casts of entity ID arrays for GPU indirect draws.
- Should `StackAllocator` be augmented with an actual circular ring buffer for the head/tail staging pattern, or is `BufferRegistry` sufficient for all GPU staging needs?

## 7. References

- `crates/renderer-backend/src/memory.rs` -- All three allocators + GpuBudget
- `engine/core/ecs/entity.py` -- Python Entity + EntityAllocator
- `gpu_driven/buffers.rs` -- Triple-buffered BufferRegistry (staging overlap)
- GAP_1_SUMMARY.md -- Investigation details for T-CORE-1.1 through T-CORE-1.5
- CLARIFICATION.md -- Rationale for divergence from original RingBuffer spec
