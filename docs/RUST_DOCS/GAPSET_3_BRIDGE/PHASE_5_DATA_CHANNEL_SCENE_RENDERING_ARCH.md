# PHASE 5: Data Channel -- Scene Rendering

**Scope:** Feed ECS component data through the Rust renderer -- mesh management, material assignment, transform upload, and indirect draw command generation.
**Depends on:** Phase 2 (ECS component store), Phase 4 (wgpu renderer)
**Produces:** MeshRegistry/MeshTable, asset_loader.rs (glTF loading), ECS-to-renderer wiring
**Status:** MINIMALLY IMPLEMENTED -- Rust MeshTable and MaterialTable exist (bindless, with WGSL companions). No glTF loader exists. No ECS-to-renderer wire-up exists. Python equivalents are fully implemented.

## 1. Overview

Phase 5 connects the ECS data channel to the renderer command channel. For each frame, the renderer needs: a list of visible meshes with their vertex/index buffers, material parameters, and per-instance transforms. The Rust implementation takes a bindless approach: `MeshTable` and `MaterialTable` are GPU-indexable arrays of mesh/material descriptors, and instances reference them by integer index. The Python engine already implements the full scene rendering pipeline -- the Rust path needs to replicate it.

## 2. Architectural decisions

- **Bindless mesh table instead of named MeshRegistry**: `gpu_driven/mesh_table.rs` implements a `MeshTable` with WGSL companion `mesh_table.wgsl`. Meshes are indexed by integer handle on the GPU, avoiding per-draw descriptor binding overhead. This differs from the original TODO's `MeshRegistry` concept but is a more modern GPU-driven approach.
- **Bindless material table**: Similarly, `gpu_driven/material_table.rs` + `material_table.wgsl` store material parameters in a GPU-accessible array. Materials are indexed by handle, and the WGSL shader samples the table at draw time.
- **glTF loading deferred to Python**: The Python asset loader (`engine/resource/asset/asset_loader.py`) handles glTF import. A Rust `asset_loader.rs` with `gltf` + `image` crates would be needed for the fully offline path, but the current architecture lets Python pre-process assets and upload the resulting buffers.
- **Indirect draw via GPU-driven approach**: `engine/rendering/gpu_driven/indirect_draw.py` exists in Python. The Rust equivalent would use `wgpu::IndirectDraw` with a GPU-generated draw command buffer, removing CPU culling bottlenecks.

## 3. Constraints specific to this phase

- Mesh data must be uploaded to GPU buffers via bytemuck Pod casts. Omega crate's Vec3 (for positions) and Vec2 (for UVs) are Pod-compatible.
- The MeshTable on the GPU is a StructuredBuffer<MeshDescriptor> -- each descriptor contains index count, index buffer offset, vertex buffer offset, and material index. Max mesh count is fixed at table creation time.
- glTF imports produce multiple meshes with multiple primitives -- the asset loader must flatten these into the MeshTable's flat indexing scheme.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `gpu_driven/mesh_table.rs` | Bindless MeshTable on GPU | EXISTS -- with WGSL companion |
| `gpu_driven/mesh_table.wgsl` | WGSL struct for mesh descriptor | EXISTS |
| `gpu_driven/material_table.rs` | Bindless MaterialTable on GPU | EXISTS -- with WGSL companion |
| `gpu_driven/material_table.wgsl` | WGSL struct for material descriptor | EXISTS |
| `gpu_driven/texture_table.rs` | Texture table management | EXISTS |
| `asset_loader.rs` | glTF/mesh loading in Rust | DOES NOT EXIST |
| `Cargo.toml` deps: gltf + image crates | Asset loading deps | NOT ADDED |
| ECS-to-renderer wiring | Read ECS components, write draw commands | DOES NOT EXIST (blocked on Phase 2 + Phase 4) |
| `engine/resource/asset/asset_loader.py` | Python glTF asset loader | EXISTS |
| `engine/rendering/gpu_driven/indirect_draw.py` | Python GPU-driven indirect draw | EXISTS |

## 5. Testing strategy

- Unit: MeshTable insert/remove/update round-trip (Rust).
- Unit: MaterialTable parameter upload (Rust).
- Unit: WGSL struct layouts match Rust struct layouts (compile-time size assertions via bytemuck).
- Integration: Load a glTF mesh in Python, upload vertices to GPU, draw via MeshTable lookup.

## 6. Open questions

- Should glTF loading be implemented in Rust (requiring `gltf` + `image` crate deps) or should Python continue to pre-process assets? Rust loading is needed for the fully offline path (no Python runtime), but Python loading is working and simpler.
- The MeshTable is fixed-size at creation. Should it support dynamic growth, or should scenes pre-allocate based on asset counts? Dynamic growth requires GPU buffer reallocation with synchronization.

## 7. References

- Phase 4 (wgpu Renderer) provides the GPU context for mesh/material upload.
- Phase 6 (PBR + Lights) consumes MaterialTable for PBR parameters.
- Phase 3 (GPU Math) supplies Vec3/Mat4 types for vertex/transform data.
- GAP_3_SUMMARY.md section "Phase 5: Scene Rendering" (5 real, 3 partial, 9 absent).
