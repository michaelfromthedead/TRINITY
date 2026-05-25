# PHASE 9: Full Features -- Post-Processing, Particles, Global Illumination

**Scope:** Implement GPU compute shaders for post-processing (tonemapping, bloom, TAA), GPU particle systems (spawn/update/render/compact), and diffuse global illumination (DDGI probe rendering and blending).
**Depends on:** Phase 6 (PBR base rendering requires post-processing to complete the frame), Phase 7 (each feature is a set of frame graph passes), Phase 10 (GPU memory for transient resources)
**Produces:** WGSL compute shaders for post-process stack, particle simulation, DDGI probe rendering
**Status:** NOT STARTED (Rust side) -- No WGSL compute shaders exist for any of these three feature areas. All three have full Python implementations. This phase concentrates the largest batch of "Python has it, Rust doesn't" gaps.

## 1. Overview

Phase 9 represents the three headline visual features that distinguish a production renderer from a tech demo. Post-processing applies final-image adjustments (tonemapping for HDR-to-LDR conversion, bloom for bright-area glow, TAA for temporal anti-aliasing). GPU particles simulate thousands of lightweight objects entirely on the GPU (compute shaders for spawn, physics update, render, and compaction). DDGI (Dynamic Diffuse Global Illumination) uses probe volumes with spherical harmonic encoding to approximate indirect light bounces. All three exist as Python prototypes but require WGSL compute shaders and frame graph integration for GPU execution.

## 2. Architectural decisions

- **Compute-shader-based post-process**: Each post-process effect (ACES tonemapping, bloom, TAA, motion blur, DOF) is a separate compute shader dispatched as a frame graph pass. The postprocess stack (`postprocess_stack.py`) chains them in order; the Rust equivalent chains wgpu compute passes.
- **GPU particle simulation with four compute passes**: Spawn (generate new particles), Update (apply forces, age, cull dead), Render (emit draw-indirect args), Compact (defragment dead slots). This is a standard GPU particle pipeline used in AAA engines.
- **DDGI probe rendering via compute**: Probe rays are traced (currently in software in Python; hardware ray tracing is deferred), SH coefficients are accumulated, and probe results are blended into the scene via a full-screen resolve pass. The Rust WGSL implementation can start with a simplified single-bounce approach.
- **Python implementations serve as reference oracles**: The Python post-process stack (10 modules), particle system (6 modules), and GI modules (2 modules) define the expected visual output. WGSL implementations must match.

## 3. Constraints specific to this phase

- TAA requires per-pixel motion vectors, which the PBR shader (Phase 6) must output. TAA cannot work without motion vector support in the main render pass.
- Bloom uses a Gaussian pyramid (downsample -> blur -> upsample), which requires multiple render targets at different scales. Frame graph resource aliasing (Phase 7) can optimize this.
- DDGI probe updates are typically spread across multiple frames to amortize cost. The compute shader must handle a frame budget (e.g., update 1/N of probes per frame).
- GPU particle counts are limited by GPU memory. The compact pass is necessary to prevent dead particles from accumulating.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| Post-process WGSL shaders | Tonemapping, bloom, TAA compute shaders | DOES NOT EXIST |
| Particle WGSL shaders | Spawn, update, render, compact computes | DOES NOT EXIST |
| DDGI WGSL shaders | Probe rendering, SH encoding, blending | DOES NOT EXIST |
| `engine/rendering/postprocess/tonemapping.py` | ACES filmic tonemapping | EXISTS (Python) |
| `engine/rendering/postprocess/bloom.py` | Gaussian bloom | EXISTS (Python) |
| `engine/rendering/postprocess/antialiasing.py` | TAA | EXISTS (Python) |
| `engine/rendering/postprocess/motion_blur.py` | Motion blur | EXISTS (Python) |
| `engine/rendering/postprocess/dof.py` | Depth of field | EXISTS (Python) |
| `engine/rendering/postprocess/color_grading.py` | Color grading LUT | EXISTS (Python) |
| `engine/rendering/postprocess/exposure.py` | Auto-exposure | EXISTS (Python) |
| `engine/rendering/postprocess/upscaling.py` | FSR/CAS upscaling | EXISTS (Python) |
| `engine/rendering/postprocess/ambient_occlusion.py` | SSAO/HBAO | EXISTS (Python) |
| `engine/rendering/postprocess/postprocess_stack.py` | Effect chain orchestrator | EXISTS (Python) |
| `engine/rendering/particles/gpu_particles.py` | GPU particle system | EXISTS (Python) |
| `engine/rendering/particles/particle_system.py` | Particle lifecycle | EXISTS (Python) |
| `engine/rendering/particles/vfx_graph.py` | Visual effects graph | EXISTS (Python) |
| `engine/rendering/particles/trail_renderer.py` | Particle trails | EXISTS (Python) |
| `engine/rendering/particles/decal_system.py` | Decal management | EXISTS (Python) |
| `engine/rendering/lighting/gi_ddgi.py` | DDGI probe simulation | EXISTS (Python) |
| `engine/rendering/lighting/gi_probes.py` | Probe management | EXISTS (Python) |

## 5. Testing strategy

- Unit: naga-compile each WGSL compute shader (bindings, workgroup size, dispatch dimensions).
- Integration: Post-process full chain -- render a scene -> tonemap -> bloom -> TAA. Compare Python and Rust output pixel-by-pixel for known test scenes.
- Integration: Particle simulation -- spawn 10,000 particles, run N frames, verify count stability (spawn = death rate at steady state).
- Integration: DDGI -- one probe in a Cornell box scene, verify irradiance approximates ground truth.

## 6. Open questions

- Should post-process effects be individual frame graph passes or combined into mega-shaders? Individual passes are easier to debug and reorder; mega-shaders reduce memory bandwidth.
- DDGI hardware ray tracing requires `wgpu::Features::RAY_TRACING`. Should the initial implementation use ray queries (WGSL) or software ray marching (compute shader)? Software ray marching works everywhere but is less accurate.
- Particle physics update runs on the GPU -- should it use a simple Euler integrator or something more sophisticated (Verlet, XPBD)? Simple Euler is standard for VFX particles.

## 7. References

- Phase 6 (PBR) provides the rendered scene that post-processing consumes.
- Phase 7 (Frame Graph) schedules each post-process/particle/DDGI pass.
- Phase 10 (GPU Memory) provides the transient buffers for post-process targets.
- GAP_3_SUMMARY.md section "Phase 9: Full Features" (5 real, 0 partial, 13 absent).
