# GAPSET_5_LIGHTING: Architectural Clarifications

## 1. Forward+ vs. Deferred: The Undecided Question

**Issue:** The TODO specifies a deferred compute shader pipeline (`lighting_pass.comp.wgsl` with G-buffer reading), but the existing WGSL implements forward PBR (`pbr.frag.wgsl`). `light_culling.wgsl` builds froxel light lists but is not consumed.

**Options:**

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| **Forward+ (with froxel culling)** | Lower bandwidth, simpler MSAA, existing `pbr.frag.wgsl` can be adapted | Need to pass froxel data to fragment shader, overdraw cost | Closest to existing code |
| **Deferred (G-buffer + compute)** | Decoupled lighting, easier many-light evaluation, HDR accumulation in compute | Higher bandwidth, MSAA complexity, no existing G-buffer pass | What TODO assumes |
| **Hybrid (deferred + forward)** | Best of both -- deferred for opaque, forward for transparent | Most complex, two shading paths | Not described anywhere |

**Recommendation:** The TODO should be updated to reflect Forward+ as the target architecture since:
- `pbr.frag.wgsl` is a forward shader and would need minimal changes
- `light_culling.wgsl` already computes froxel assignments
- The shader `light_culling.wgsl` outputs froxel light lists that can be read by a fragment shader via storage buffer
- No G-buffer pass, GBuffer struct, or render target management exists

However, this changes the acceptance criteria for most Phase 3 tasks (they reference a deferred compute pipeline that should become a forward+ pipeline).

## 2. Python Python Reference vs. WGSL Reality

**Issue:** The Python modules in `engine/rendering/lighting/` implement a complete reference design but have no runtime connection to the WGSL shaders.

**Python modules (implement all 7 types + all techniques):**
- `light_types.py`: DirectionalLight, PointLight, SpotLight, RectAreaLight, DiskAreaLight, IESLight, SkyLight
- `light_culling.py`: Proper frustum AABB reconstruction, all light intersection types
- `shadows.py`: CSM with texel snapping, CubeShadowMap, SpotShadowMap, ShadowAtlas bin-packing
- `shadow_filtering.py`: PCF, PCSS, VSM, ESM, ContactShadows
- `gi_ddgi.py`: DDGI with octahedral encoding, Fibonacci ray distributions, temporal hysteresis
- `gi_probes.py`: SH L2, ProbeGrid, IrradianceVolume, Lightmaps, ReflectionProbes

**WGSL implements:**
- 3 light types (dir/point/spot)
- CSM with PCF (basic grid kernel)
- DDGI compute (simplified ray marching)
- Forward PBR (Cook-Torrance BRDF)

**The bridge that connects them does not exist.** The GAPSET_3_BRIDGE analysis identified that the `_omega` PyO3 module was never built, so Python and WGSL/Rust are entirely independent codebases. The Python reference is the design specification; the WGSL shaders are a partial, independent implementation.

**Recommendation:** For each shader task, the Python module is the reference implementation. Acceptance criteria should verify WGSL output against Python computation, not against external papers or measurements.

## 3. Cross-References to GAPSET_3_BRIDGE

GAPSET_3_BRIDGE built the shared rendering infrastructure that GAPSET_5 depends on:

| GAP 3 Artifact | Type | Location | Status |
|----------------|------|----------|--------|
| Light culling WGSL | Compute shader | `shaders/light_culling.wgsl` | REAL (229 lines) |
| CSM WGSL | Functions | `shaders/shadow_csm.wgsl` | REAL (161 lines) |
| DDGI WGSL | Compute shaders | `shaders/ddgi.wgsl` | REAL (240 lines) |
| PBR vertex | Vertex shader | `shaders/pbr.vert.wgsl` | REAL (64 lines) |
| PBR fragment | Fragment shader | `shaders/pbr.frag.wgsl` | REAL (377 lines) |
| Shadow vertex | Vertex shader | `shaders/shadow.vert.wgsl` | REAL (35 lines) |
| Shadow fragment | Fragment shader | `shaders/shadow.frag.wgsl` | REAL (12 lines) |
| Material table WGSL | Buffer defs | `src/gpu_driven/material_table.wgsl` | REAL (97 lines) |
| DDGI Rust | Pass builder | `src/ddgi.rs` | REAL (303 lines) |
| Frame graph IR | Compiler | `src/frame_graph/mod.rs` | REAL (108KB) |
| GPU buffers | Staging | `src/gpu_driven/buffers.rs` | REAL (24KB) |
| Renderer skeleton | Runtime | `src/renderer.rs` | PARTIAL (triangle only) |

**Dependency chain for GAP 5 tasks:**
- T-LIT-3.1 depends on `S3 material_shared.wgsl` -- this does not exist (no shared WGSL math module). BRDF functions are inlined in `pbr.frag.wgsl`.
- T-LIT-1.4 depends on `S14 RHI` -- no wgpu runtime binding layer exists (the renderer.rs is a triangle demo).
- T-LIT-1.1 depends on `S15 math.rs` -- no Rust math library exists in the renderer-backend crate.
- T-LIT-3.5 depends on `S4-G4 (IES parser)` -- no IES parser exists in Rust.

## 4. Physical Units Convention

The Python reference uses physically-based units:
- **DirectionalLight**: intensity in lux (lm/m^2)
- **PointLight**: intensity in lumens, attenuation `saturate(1 - (d/r)^4)^2 / (d^2 + 1)`
- **SpotLight**: intensity in candelas, angular smoothstep + distance attenuation
- **RectAreaLight/DiskAreaLight**: intensity in nits (cd/m^2)
- **IESLight**: candela values from IES profile

The WGSL in `pbr.frag.wgsl` uses a simpler `smooth attenuation = pow(clamp(1 - d^2/r^2, 0, 1), 2)` formula applied uniformly. This is physically plausible but not unit-accurate. The TODO should specify whether physical unit accuracy is required or if plausible-to-the-eye is sufficient.

## 5. Shadow Map Resolution Strategy

**TODO assumption:** 2D atlas with 4 tile sizes (256-2048), 4096 atlas resolution.

**Reality:** `shadow_csm.wgsl` and `pbr.frag.wgsl` hardcode `SHADOW_MAP_RESOLUTION = 2048.0`. Atlas is in Python only.

The GAP 5 should decide between:
1. **Array textures** (current approach) -- simple, per-cascade independent resolution, limited by array layer count. Good for CSM (4 layers). Bad for many shadow-casting lights.
2. **Shadow atlas** (TODO approach) -- complex packing, any number of lights, requires atlas UV transforms, efficient for VRAM. Required for cube + spot shadows.

Both strategies can coexist: array textures for CSM, atlas for point/spot shadows.

## 6. DDGI Implementation Status

The DDGI implementation is the most complete bridge component:
- `ddgi.wgsl`: Two compute entry points (update + sample), SH L0+L1 evaluation, probe grid indexing, trilinear interpolation, temporal hysteresis with blend alpha = 0.3
- `ddgi.rs`: Frame graph pass builder with `DDGIProbeVolume` descriptor and test suite (22 tests)
- Python `gi_ddgi.py`: Complete reference with octahedral encoding, Fibonacci spiral ray directions, Chebyshev visibility weighting, hysteresis accumulation

**Gaps:**
- WGSL uses simplified single-step ray marching (placeholder)
- WGSL uses SH L0+L1 only (Python uses octahedral encoding)
- WGSL has no Chebyshev visibility weighting (Python has full implementation)
- The Rust pass builder has no runtime execution -- it constructs `IrPass` objects but no wgpu compute pipeline exists to run them
- Frame graph integration is incomplete -- passes are created but not scheduled in a pipeline

## 7. Task Count Discrepancy

The TODO header claims 49 total tasks but only 33 are specified across 6 phases. The missing 16 tasks would be for Phases 7-9 which are not written. The 33 specified tasks are the actionable scope.
