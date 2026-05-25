# GAPSET 8: GPU Compute -- Reality Documentation Check Summary

> **RDC Date**: 2026-05-22
> **Total tasks**: 32 (35 listed in summary table, 32 unique task IDs)
> **Reality verdict**: 11 [x] (implemented), 10 [~] (partial/alternative), 11 [-] (not implemented)

---

## Verdict By Phase

| Phase | [x] | [~] | [-] | Reality Score |
|-------|-----|-----|-----|---------------|
| 1: Foundation (6 tasks) | 6 | 0 | 0 | **100%** |
| 2: GPU Compute Core (4 tasks) | 0 | 2 | 2 | **25%** |
| 3: Core Culling (4 tasks) | 0 | 1 | 3 | **12%** |
| 4: Occlusion + Meshlet (5 tasks) | 0 | 1 | 4 | **10%** |
| 5: Particle Compute (4 tasks) | 3 | 1 | 0 | **88%** |
| 6: Particle/VFX Render (5 tasks) | 3 | 1 | 1 | **70%** |
| 7: Integration (4 tasks counted, actually 7) | 0 | 5 | 2 | **28%** |

## Critical Findings

### T-GPU-1.4 (Material Table) and T-GPU-1.5 (Texture Table) are FULLY IMPLEMENTED but marked [ ] in TODO
Both material_table.rs (1302 lines) and texture_table.rs (271 lines) exist with full stage/dirty-tracking/free-list implementations including WGSL counterparts. Checkboxes should be [x].

### T-GPU-5.1, 5.2, 5.3 (Particle Spawn/Update/Compact) are IMPLEMENTED in particles.wgsl
All three compute entry points exist plus a reset entry point. The compact is swap-based (not prefix-sum as specified), but functionally present. Checkboxes should be [x].

### T-GPU-6.3 (Trail Renderer), 6.4 (Decal System), 6.5 (VFX Graph) are FULLY IMPLEMENTED
trail_renderer.py (815 lines), decal_system.py (946 lines), vfx_graph.py (946 lines) all exist with full class hierarchies. Checkboxes should be [x].

### No shaders/gpu_driven/ or shaders/common/ directories exist
All GPU compute shaders referenced for Phases 2-4 (radix sort, compaction, culling, HZB) do not exist as files. This is consistent with those tasks remaining unimplemented.

### Key file location issues
- TODO references `crates/.../gpu_driven/bindless.rs` for mesh/material/texture tables -- actual files are `mesh_table.rs`, `material_table.rs`, `texture_table.rs` in the same directory
- TODO references `crates/.../gpu_driven/indirect_draw.rs` -- only Python side exists at `engine/rendering/gpu_driven/indirect_draw.py`
- TODO references `trinity/decorators/gpu.py` for `@gpu_driven_mesh` -- this composite decorator does not exist anywhere

### Phase 7 task count discrepancy
Summary table says 7 tasks for Phase 7 (T-GPU-7.1 through T-GPU-7.7), but the task body lists 7 tasks -- wait, the body actually lists 7 tasks (T-GPU-7.1 through T-GPU-7.7), and the summary table says 7. Someone else said "4 tasks counted" in their analysis; rechecking: the body has T-GPU-7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7 which is 7. Summary table says 7. That's consistent. Wait, the header of this file says "4 tasks counted" -- let me recount. Body: T-GPU-7.1, T-GPU-7.2, T-GPU-7.3, T-GPU-7.4, T-GPU-7.5, T-GPU-7.6, T-GPU-7.7. Yes, 7 tasks.

## File Existence Matrix

| File | Exists | Lines | Status |
|------|--------|-------|--------|
| `crates/renderer-backend/src/gpu_driven/buffers.rs` | YES | 777 | Complete |
| `crates/renderer-backend/src/gpu_driven/material_table.rs` | YES | 1302 | Complete |
| `crates/renderer-backend/src/gpu_driven/material_table.wgsl` | YES | 97 | Complete |
| `crates/renderer-backend/src/gpu_driven/mesh_table.rs` | YES | 1803 | Complete |
| `crates/renderer-backend/src/gpu_driven/mesh_table.wgsl` | YES | 63 | Complete |
| `crates/renderer-backend/src/gpu_driven/texture_table.rs` | YES | 271 | Complete |
| `crates/renderer-backend/src/gpu_driven/mod.rs` | YES | 40 | Exports all tables |
| `crates/renderer-backend/src/particles.rs` | YES | 431 | Complete |
| `crates/renderer-backend/src/memory.rs` | YES | 452 | 4 allocators |
| `crates/renderer-backend/shaders/particles.wgsl` | YES | 272 | Spawn/Update/Render/Compact |
| `crates/renderer-backend/shaders/light_culling.wgsl` | YES | 230 | Forward+ culling |
| `crates/renderer-backend/shaders/ddgi.wgsl` | YES | 241 | DDGI probes |
| `crates/renderer-backend/shaders/pbr.frag.wgsl` | YES | 378 | PBR fragment |
| `crates/renderer-backend/shaders/pbr.vert.wgsl` | YES | 57 | PBR vertex |
| `crates/renderer-backend/shaders/shadow_csm.wgsl` | YES | 166 | CSM shadow |
| `trinity/decorators/gpu.py` | YES | 887 | 8 decorators |
| `trinity/decorators/rendering.py` | YES | 443 | 6 decorators |
| `trinity/decorators/lod_streaming.py` | YES | 292 | 5 decorators |
| `trinity/decorators/particles_vfx.py` | YES | N/A | @gpu_particle decorator |
| `engine/rendering/gpu_driven/culling.py` | YES | 1109 | CPU culling pipeline |
| `engine/rendering/gpu_driven/indirect_draw.py` | YES | 661 | CPU indirect draw |
| `engine/rendering/gpu_driven/bindless.py` | YES | 786 | CPU bindless mgmt |
| `engine/rendering/gpu_driven/meshlet.py` | YES | 731 | CPU meshlet system |
| `engine/rendering/gpu_driven/visibility_buffer.py` | YES | 836 | CPU visibility buffer |
| `engine/rendering/gpu_driven/instancing.py` | YES | 736 | CPU instancing |
| `engine/rendering/particles/trail_renderer.py` | YES | 815 | Trail system |
| `engine/rendering/particles/decal_system.py` | YES | 946 | Decal system |
| `engine/rendering/particles/vfx_graph.py` | YES | 946 | VFX graph |
| `engine/rendering/particles/gpu_particles.py` | YES | 776 | GPU sim framework |
| `engine/rendering/particles/particle_system.py` | YES | 855 | CPU particle system |
| `engine/rendering/particles/particle_modules.py` | YES | 1060 | Particle modules |
| `shaders/gpu_driven/*` | NO | -- | Directory does not exist |
| `shaders/common/*` | NO | -- | Directory does not exist |
| `crates/.../gpu_driven/sort.rs` | NO | -- | Not implemented |
| `crates/.../gpu_driven/indirect_draw.rs` | NO | -- | Only Python impl |
| `crates/.../gpu_driven/hzb.rs` | NO | -- | Not implemented |
| `crates/.../gpu_driven/meshlet_culling.rs` | NO | -- | Not implemented |
| `crates/.../gpu_driven/lod_feedback.rs` | NO | -- | Not implemented |
| `crates/.../gpu_driven/fallback.rs` | NO | -- | Not implemented |

## Summary

Foundation (Phase 1) is 100% complete with all four tables (buffer, mesh, material, texture) fully implemented in Rust with WGSL counterparts. Particle compute pipeline (Phase 5) is 88% complete with spawn/update/compact/render entry points in WGSL plus Rust pass factories. Particle/VFX rendering (Phase 6) is 70% complete with trail, decal, and VFX graph subsystems.

GPU compute core (Phase 2), culling pipeline (Phase 3), and occlusion/meshlet pipeline (Phase 4) are substantially unimplemented at the GPU level. Integration (Phase 7) has Python-side decorators but no Rust-side dispatch or GPU feedback loops.

**8 checkbox errors in PHASE_N_TODO.md**: T-GPU-1.4, T-GPU-1.5, T-GPU-5.1, T-GPU-5.2, T-GPU-5.3, T-GPU-6.3, T-GPU-6.4, T-GPU-6.5 are all implemented but marked [ ].
