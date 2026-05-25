# PHASE 10: GPU Memory Management

**Scope:** Manage GPU memory allocation, deallocation, and streaming -- including transient resource pools, mipmap streaming, and budget enforcement.
**Depends on:** Phase 4 (wgpu device for buffer/texture creation)
**Produces:** GpuMemoryManager in Rust (with sub-allocators: linear, pool, ring, slab, stack, TLSF), streaming resource pool with LRU eviction
**Status:** NOT STARTED (Rust side) -- No Rust GpuMemoryManager exists. The Python memory subsystem (`engine/core/memory/`) is complete with 10 allocator types. Python resource streaming (`engine/resource/streaming/` and `engine/resource/memory/`) is also complete. These are the most portable and testable of the Python subsystems.

## 1. Overview

GPU memory management is the infrastructure layer that makes all other phases efficient. Without it, every mesh upload requires a new `wgpu::Buffer`, every render target allocates fresh memory, and there is no mechanism to stream textures in/out based on camera distance. The Python memory subsystem proves the allocator designs (linear, pool, ring, slab, stack, TLSF, object_pool, tracker) -- these are well-known algorithms that map directly to wgpu buffer sub-allocation. The streaming system handles LRU eviction, mipmap prioritization, and budget management.

## 2. Architectural decisions

- **GPU sub-allocation via wgpu::BufferDeviceAddress**: Rather than creating many small wgpu buffers (which hits driver allocation overhead), allocate large GPU buffers and sub-allocate from them. DeviceAddress enables GPU-side pointer arithmetic.
- **Multiple allocator strategies for different resource types**: Linear allocators for per-frame transient data (upload ring buffer), pool allocators for constant-size resources (mesh descriptors), TLSF for general-purpose variable-size allocations. Each maps to a specific `wgpu::BufferUsage` pattern.
- **Streaming with LRU eviction**: The streaming system tracks resource usage frequency and evicts least-recently-used resources when GPU memory budget is exceeded. Mipmap streaming loads only the mip levels needed for the current camera distance.
- **Budget management with dynamic priority**: `engine/resource/memory/budget_manager.py` assigns priority scores to resources. When budget is exceeded, the lowest-priority resources are evicted first.

## 3. Constraints specific to this phase

- wgpu buffer alignment requirements (typically 256 bytes for structured buffers, 16 bytes for constant buffers) must be respected by all sub-allocators.
- GPU memory cannot be paged out to disk -- evicted resources must be fully reloaded from source assets.
- Streaming decisions must be asynchronous: initiate load in frame N, complete by frame N+K (K = latency budget in frames).
- Budget enforcement must never block the render thread -- eviction and streaming are background operations.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `memory.rs` | Rust GpuMemoryManager | DOES NOT EXIST |
| Rust LinearAllocator | Per-frame bump allocator | DOES NOT EXIST |
| Rust PoolAllocator | Constant-size resource allocator | DOES NOT EXIST |
| Rust TLSFAllocator | Variable-size general allocator | DOES NOT EXIST |
| Rust LRU eviction | Least-recently-used eviction policy | DOES NOT EXIST |
| Rust mipmap streaming | Distance-based mip loading | DOES NOT EXIST |
| `engine/core/memory/allocator.py` | Abstract allocator base | EXISTS |
| `engine/core/memory/linear.py` | Linear bump allocator | EXISTS |
| `engine/core/memory/pool.py` | Fixed-size pool allocator | EXISTS |
| `engine/core/memory/ring.py` | Ring buffer allocator | EXISTS |
| `engine/core/memory/slab.py` | Slab allocator | EXISTS |
| `engine/core/memory/stack.py` | Stack allocator | EXISTS |
| `engine/core/memory/tlsf.py` | TLSF allocator | EXISTS |
| `engine/core/memory/tracker.py` | Allocation tracking | EXISTS |
| `engine/core/memory/object_pool.py` | Object pool | EXISTS |
| `engine/resource/streaming/mesh_streaming.py` | Mesh streaming | EXISTS |
| `engine/resource/streaming/texture_streaming.py` | Texture streaming | EXISTS |
| `engine/resource/streaming/stream_manager.py` | Streaming orchestrator | EXISTS |
| `engine/resource/streaming/priority_system.py` | Load priority system | EXISTS |
| `engine/resource/memory/budget_manager.py` | GPU budget enforcement | EXISTS |
| `engine/resource/memory/eviction.py` | LRU eviction policy | EXISTS |
| `engine/resource/memory/residency_manager.py` | Residency tracking | EXISTS |
| `engine/resource/memory/asset_pool.py` | Asset memory pool | EXISTS |

## 5. Testing strategy

- Unit: Each Rust allocator tested in isolation (allocate -> write -> read -> deallocate -> no corruption).
- Unit: LRU eviction policy test -- fill budget, add new resource, verify lowest-priority resource is evicted.
- Unit: Sub-allocation alignment -- verify all allocations satisfy wgpu alignment requirements.
- Integration: Stream a high-resolution texture through the mipmap pipeline -- verify only needed mips are loaded.
- Integration: Budget enforcement -- exceed GPU memory budget, verify streaming system evicts and recovers.

## 6. Open questions

- Should the Rust allocators be a direct port of the Python implementations (proven correctness) or a new implementation optimized for wgpu? Direct port minimizes risk; wgpu optimization can follow.
- Is wgpu `BufferDeviceAddress` available on all targets? It requires the `wgpu::Features::BUFFER_DEVICE_ADDRESS` feature, which is available on Vulkan and DX12 but not WebGPU. A fallback path (using bind group offsets) is needed for WebGPU targets.
- The Python memory tracker is useful for debugging. Should it be ported to Rust, or should it remain a Python-only debugging tool? Remaining in Python avoids polluting release builds with tracking overhead.

## 7. References

- Phase 4 (wgpu Renderer) creates the device that owns GPU memory.
- Phase 9 (Full Features) uses transient buffers for post-process targets and particle data.
- GAP_3_SUMMARY.md section "Phase 10: GPU Memory Management" (4 real, 0 partial, 8 absent).
