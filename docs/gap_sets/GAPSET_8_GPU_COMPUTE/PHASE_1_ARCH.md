# PHASE 1 ARCHITECTURE: Foundation -- Buffer Management & Bindless Infrastructure

> **Phase**: 1/7 | **Status**: [x] 100% Complete
> **Tasks**: T-GPU-1.1 through T-GPU-1.6 (6 tasks)
> **Gaps**: S2-G2, S2-G4 (partial), S2-G5

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `trinity/decorators/gpu.py` | 887 | @gpu_struct, @gpu_buffer, @bind_group decorators |
| `crates/.../gpu_driven/buffers.rs` | 777 | BufferRegistry with triple-buffered staging |
| `crates/.../gpu_driven/mesh_table.rs` | 1803 | Bindless Mesh Table |
| `crates/.../gpu_driven/mesh_table.wgsl` | 63 | GPU mesh entry + helpers |
| `crates/.../gpu_driven/material_table.rs` | 1302 | Bindless Material Table |
| `crates/.../gpu_driven/material_table.wgsl` | 97 | GPU material entry + helpers |
| `crates/.../gpu_driven/texture_table.rs` | 271 | Bindless Texture Table |
| `crates/.../gpu_driven/mod.rs` | 40 | Module exports |

## Architecture

### BufferRegistry Triple-Buffered Staging (buffers.rs)
```
Frame N:   CPU writes → Slot[write_index] (Writing state)
Frame N-1: GPU reads  → Slot[read_index]  (Reading state)
Frame N-2: Slot being reclaimed           (Free state)

SlotState machine: Free → acquire() → Writing → submit() → Ready → start_read() → Reading → release() → Free
```

Key design: round-robin write_index probe, monotonically increasing frame_count, `acquire_reading()` returns newest Ready slot by frame_index. Back-pressure when all 3 slots occupied (`is_stalled()`).

### Bindless Tables (mesh_table.rs, material_table.rs, texture_table.rs)
All three tables share a common pattern:
- **MeshTableEntry** (24 bytes, 6 x u32): index_offset, vertex_offset, index_count, vertex_count, material_id, flags
- **MaterialTableEntry** (80 bytes, repr(C), align(16)): base_color, emissive, metallic, roughness, occlusion, normal_scale, texture IDs, flags (bit 0=visible, bit 31=dirty), alpha_cutoff
- **TextureTableEntry** (24 bytes, 6 x u32): width, height, mip_levels, format, layer_count, flags

Each table supports:
- `add()` / `remove()` / `update()` — CPU-side management
- `stage()` → acquire slot from BufferRegistry, copy bytes
- `stage_and_submit()` → stage + submit in one call
- MaterialTable has dirty-flag tracking (`any_dirty()`, `mark_dirty()`, `clear_dirty()`)

### @gpu_struct Decorator (gpu.py)
- Vec2 (8 bytes, align 8), Vec3 (12 bytes, align 16), Vec4 (16 bytes, align 16), Mat4 (64 bytes, align 16)
- f32[N] via `Annotated[float, N]` (fixed-size arrays)
- Nested @gpu_struct classes
- WGSL storage-buffer alignment rules: element_stride = roundUp(alignof(T), sizeof(T)), array alignment = max(16, align(T))

### @gpu_buffer Decorator (gpu.py)
- Usage flags: vertex, index, uniform, storage, indirect, map_read, map_write, copy_src, copy_dst
- Auto-appends COPY_DST for storage/indirect (WebGPU spec requirement)
- `_resolve_wgpu_usage_flags()` produces wgpu bitmask
- `create_wgpu_buffer()` convenience function

### @bind_group Decorator (gpu.py)
- index validation (must be >= 0)
- Tags class with bind_group_index

## Verification

All 6 tasks verified [x] from source code. T-GPU-1.4 and T-GPU-1.5 were marked [ ] in TODO but are fully implemented.
