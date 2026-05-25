# PHASE 6: Data Channel -- PBR + Lights

**Scope:** Implement physically-based shading (Cook-Torrance BRDF), forward+ light culling, and shadow mapping in WGSL, with material parameters fed through the Rust material table and light data uploaded from the ECS.
**Depends on:** Phase 4 (wgpu renderer), Phase 5 (mesh/material tables for rendering)
**Produces:** WGSL PBR shaders (pbr.vert/frag, shadow vert/frag), forward+ light culling compute shader, PipelineTable + ShaderCache
**Status:** NOT STARTED (Rust side) -- No WGSL PBR, lighting, or shadow shaders exist. The Rust material_table.rs provides bindless material management infrastructure but does not compile shaders. All PBR/lighting/shadow logic exists in Python.

## 1. Overview

Phase 6 is the rendering quality inflection point: it moves from "see something on screen" (Phase 4's triangle) to "see something that looks good." The Cook-Torrance BRDF handles diffuse (Lambertian) + specular (GGX) + Fresnel (Schlick) terms. Forward+ light culling partitions the screen into froxels and assigns lights per froxel via a compute shader. Cascaded shadow maps (CSM) provide directional shadowing. The Rust material_table.rs provides the infrastructure for bindless material indexing on the GPU, but the actual PBR parameter interpretation and shading logic must be implemented in WGSL.

## 2. Architectural decisions

- **Bindless material parameters**: MaterialTable (gpu_driven) stores PBR parameters (albedo, metallic, roughness, ambient occlusion, emissive) in a GPU StructuredBuffer. Shaders sample by material_index rather than binding per-material descriptors.
- **Forward+ rendering, not deferred**: Forward+ avoids the G-buffer memory bandwidth cost and simplifies MSAA. Light culling happens in a compute shader before the main draw pass.
- **Python BRDF as reference implementation**: `engine/rendering/materials/pbr_model.py` implements the full Cook-Torrance BRDF. WGSL shaders should reproduce this output exactly -- the Python version serves as a validation oracle.
- **PipelineTable as compilation cache**: The intended (but unimplemented) PipelineTable caches wgpu RenderPipeline objects keyed by shader variant hash. Creating a pipeline in wgpu is expensive -- the table avoids repeated compilation.

## 3. Constraints specific to this phase

- All WGSL shaders must be validatable by naga 24 at compile time (via include_str! in tests).
- PBR shaders need access to MaterialTable (bindless), light buffer (froxel light list), shadow map texture array, and camera uniforms (Mat4 view/projection).
- Shadow maps require a separate render pass with depth-only output, which means either a second pipeline or a multi-pass frame graph.
- Forward+ froxel grid dimensions must match the tile size used in the compute shader (typically 16x16 pixels).

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `shaders/pbr.vert.wgsl` | PBR vertex shader | DOES NOT EXIST |
| `shaders/pbr.frag.wgsl` | PBR fragment shader (Cook-Torrance BRDF) | DOES NOT EXIST |
| `shaders/shadow.vert.wgsl` | Shadow map vertex shader | DOES NOT EXIST |
| `shaders/shadow.frag.wgsl` | Shadow map fragment shader (depth-only) | DOES NOT EXIST |
| `pipeline.rs` | PipelineTable + ShaderCache | DOES NOT EXIST |
| Forward+ compute shader | Froxel light culling | DOES NOT EXIST |
| `gpu_driven/material_table.rs` | Bindless material parameter storage | EXISTS -- management layer, no shader compilation |
| `gpu_driven/material_table.wgsl` | Material descriptor WGSL struct | EXISTS |
| `engine/rendering/materials/pbr_model.py` | Python Cook-Torrance BRDF | EXISTS |
| `engine/rendering/lighting/light_culling.py` | Python forward+ light culling | EXISTS |
| `engine/rendering/lighting/shadows.py` | Python CSM shadow maps | EXISTS |
| `engine/rendering/lighting/shadow_filtering.py` | Python PCF/PCSS shadow filtering | EXISTS |

## 5. Testing strategy

- Unit: naga-compile each WGSL shader at test time (include_str! validation).
- Unit: Python vs Rust BRDF output comparison for known inputs (albedo, normal, light direction, metallic, roughness).
- Integration: Render a PBR sphere with known material parameters, verify pixel output matches Python reference.
- Integration: Forward+ culling test -- N lights in scene, verify each froxel's light list is correct.

## 6. Open questions

- Should shadow maps use CSM (cascaded), PSSM (parallel-split), or a single shadow map? Python shadows.py implements CSM with configurable cascade splits. The WGSL port should match.
- PipelineTable is a wgpu-facing concept. Should it live in a new `pipeline.rs` or be part of the existing `gpu_driven/` module? It has no GPU-driven logic -- it is purely about pipeline caching.
- The T-BRG-6.3 forward+ compute shader and T-BRG-6.4 shadow map shaders could be parallel work streams. Both depend only on Phase 4 existing (which it does not).

## 7. References

- Phase 5 (Scene Rendering) provides the MeshTable for PBR draw calls.
- Phase 3 (GPU Math) provides Vec3/Mat4 types for light/direction math.
- Phase 7 (Frame Graph) schedules PBR and shadow passes.
- GAP_3_SUMMARY.md section "Phase 6: PBR + Lights" (3 real, 2 partial, 15 absent).
