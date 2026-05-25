# PHASE 2 ARCHITECTURE: GPU Compute Core -- Radix Sort & Compaction

> **Phase**: 2/7 | **Status**: [~] 25% (2 partial, 2 absent)
> **Tasks**: T-GPU-2.1 through T-GPU-2.4 (4 tasks)
> **Gaps**: S2-G6, S2-G1 (compaction), S9-G3 (compact), S9-G4 (sort)

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `crates/.../shaders/particles.wgsl` (particle_compact) | 31 lines (235-265) | Swap-based compaction (not prefix-sum) |
| `engine/rendering/gpu_driven/indirect_draw.py` | 661 | CPU-side indirect draw structures |

## NOT Implemented

| File | Status |
|------|--------|
| `shaders/gpu_driven/gpu_sort.comp.wgsl` | [-] Does not exist |
| `crates/.../gpu_driven/sort.rs` | [-] Does not exist |
| `shaders/gpu_driven/gpu_compact.comp.wgsl` | [-] Does not exist (particle compaction in particles.wgsl, not gpu_driven/) |
| `shaders/common/prefix_sum.wgsl` | [-] Does not exist, shaders/common/ does not exist |
| `crates/.../gpu_driven/indirect_draw.rs` | [-] Does not exist |

## Reality by Task

### T-GPU-2.1: GPU radix sort [ - ] NOT IMPLEMENTED
No gpu_sort.comp.wgsl exists in the project. No sort.rs exists. No WGSL radix sort shader of any kind.

### T-GPU-2.2: Buffer compaction [ ~ ] PARTIAL
A compaction compute shader exists in `particles.wgsl` (`particle_compact()`), but it is a simplified **swap-based** approach, NOT prefix-sum + scatter:
- Scans alive particles, swaps dead ones with the last alive particle
- Uses racy atomic decrement (`atomicSub(&alive_count[0], 1u)`)
- Documented limitation: "one-frame visual glitch at worst" and "racy but converging"
- No proper prefix-sum utility, no dedicated compact buffer with prefix-sum pass

### T-GPU-2.3: Shared prefix-sum [ - ] NOT IMPLEMENTED
No `shaders/common/` directory exists anywhere in the project. No prefix_sum.wgsl exists.

### T-GPU-2.4: Indirect draw buffers [ ~ ] PARTIAL
CPU-side indirect draw structures fully implemented in Python:
- `DrawIndexedIndirectArgs`, `DrawIndirectArgs` with `to_bytes()`/`from_bytes()`
- `IndirectDrawBuffer` manager, `MultiDrawBatch`
- No Rust-side GPU buffer allocation (`indirect_draw.rs`)
- No wgpu buffer with INDIRECT | STORAGE usage as specified

## Code Details

### particle_compact() algorithm (existing, particles.wgsl:235-265)
```
for each thread gid.x < alive_count:
    if particle[gid.x].lifetime <= 0:
        last_idx = alive_count - 1
        if gid.x < last_idx and particle[last_idx] is alive:
            swap particle[gid.x] with particle[last_idx]
        atomicSub(&alive_count[0], 1)
```
This is O(n) per frame but may temporarily leave gaps. The `particle_reset_dead_count()` single-thread shader resets the dead counter.

### Recommended implementation path for T-GPU-2.1 (radix sort):
```
8-pass radix sort (32-bit key, 4-bit radix):
  Pass 1-8: histogram digit → prefix sum → scatter
  Requires: counters buffer, temp buffer, prefix_sum utility → shaders/common/prefix_sum.wgsl
```

### Recommended implementation path for T-GPU-2.2 (proper compaction):
```
Prefix-sum compaction:
  1. Compute prefix sum over alive flags → output offsets
  2. Scatter alive entries to new buffer using prefix-sum offsets
  3. Write alive count to indirect draw counter buffer
  Reuses: prefix_sum.wgsl from T-GPU-2.3
```
