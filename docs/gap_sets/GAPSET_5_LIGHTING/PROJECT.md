# GAPSET_5_LIGHTING: Lighting Pipeline

## Scope

Build the GPU lighting pipeline for the TRINITY rendering engine, covering light type infrastructure, clustered light culling, PBR shading, shadow maps (CSM, cube, spot), shadow atlas management, and global illumination (DDGI, light probes).

## Current State

The project has substantial Python reference implementations and partial WGSL shaders, but **no integrated GPU lighting pipeline**. The existing forward PBR shader (`pbr.frag.wgsl`) is functional for 3 light types with CSM shadows, but the deferred froxel-culled architecture envisioned in the phase plan was never built.

### What Exists

| Layer | Location | Lines | Status |
|-------|----------|-------|--------|
| Python light types | `engine/rendering/lighting/light_types.py` | 650 | Full reference (7 types) |
| Python light culling | `engine/rendering/lighting/light_culling.py` | 619 | Full reference (CPU) |
| Python shadow maps | `engine/rendering/lighting/shadows.py` | 785 | Full reference (CSM + cube + spot + atlas) |
| Python shadow filters | `engine/rendering/lighting/shadow_filtering.py` | 797 | Full reference (5 techniques) |
| Python DDGI | `engine/rendering/lighting/gi_ddgi.py` | 844 | Full reference |
| Python light probes | `engine/rendering/lighting/gi_probes.py` | 779 | Full reference |
| Python constants | `engine/rendering/lighting/constants.py` | 175 | Reference constants |
| WGSL light culling | `shaders/light_culling.wgsl` | 229 | Unconsumed (forward+ not wired) |
| WGSL CSM sampling | `shaders/shadow_csm.wgsl` | 161 | Used by pbr.frag.wgsl |
| WGSL DDGI | `shaders/ddgi.wgsl` | 240 | Full compute (update + sample) |
| WGSL PBR forward | `shaders/pbr.frag.wgsl` | 377 | Main lighting shader (3 types) |
| WGSL shadow depth | `shaders/shadow.vert/frag.wgsl` | 47 | CSM depth-only rendering |
| Rust DDGI frames | `crates/renderer-backend/src/ddgi.rs` | 303 | Frame graph pass builders |

### What Must Be Built

1. **Rust light type definitions** -- `light_types.rs` with repr(C) GPU structs for all 7 types
2. **Light data upload** -- CPU-to-GPU staging with SoA/AoS conversion
3. **Froxel culling integration** -- Wire `light_culling.wgsl` into the pipeline, add proper AABBs, atomics
4. **Forward+ or deferred path decision** -- Architect the shading pipeline (forward+ with froxel-culled lights)
5. **Missing light types in WGSL** -- Area (LTC), IES, Sky light evaluation
6. **Missing shadow shaders** -- Cube shadow WGSL, spot shadow WGSL
7. **Shadow atlas implementation** -- WGSL shadow_common.wgsl, shadow_filter_pcf/pcss.wgsl
8. **Rust dispatch modules** -- `culling.rs`, `lighting_pass.rs`, `csm.rs`, `cube.rs`, `spot.rs`, `atlas.rs`
9. **DDGI integration** -- Wire `ddgi.wgsl` update/sample passes into frame graph
10. **GPU correctness tests** -- Readback tests comparing GPU output to Python reference

## Goals

1. **Complete the Python-to-WGSL bridge** such that the Python reference modules produce byte-identical GPU data
2. **Integrate all 7 light types** into WGSL shaders with proper physical units
3. **Implement forward+ or deferred clustered rendering** with froxel-culled light lists
4. **Build complete shadow pipeline** including CSM, cube shadow, spot shadow, atlas, PCF, PCSS
5. **Wire DDGI** into the frame graph for indirect diffuse lighting
6. **Add GPU correctness tests** comparing against Python CPU reference

## Key Constraints

- Python reference modules must remain the source of truth for algorithm validation
- WGSL shaders must match GPU structure layout with the Rust-side definitions
- Frame graph IR (`crates/renderer-backend/src/frame_graph/`) is the integration mechanism for all passes
- No external ray tracing hardware dependency -- all DDGI is compute-shader based

## Related Docs

- `PHASE_N_TODO.md` -- Task inventory with RDC corrections
- `GAP_5_SUMMARY.md` -- Full verification report
- `CLARIFICATION.md` -- Architectural philosophy and design decisions
- `PHASE_1_LIGHT_DATA_ARCH.md` -- GPU Light Data Infrastructure architecture
- `PHASE_2_FROXEL_ARCH.md` -- Froxel Clustered Culling architecture
- `PHASE_3_PBR_ARCH.md` -- PBR Lighting architecture
- `PHASE_4_CSM_ARCH.md` -- Cascaded Shadow Maps architecture
- `PHASE_5_CUBE_SPOT_ARCH.md` -- Cube + Spot Shadow Maps architecture
- `PHASE_6_SHADOW_ATLAS_ARCH.md` -- Shadow Atlas + Filtering architecture

## Cross-References

- GAPSET_3_BRIDGE: Shared WGSL infrastructure (light_culling.wgsl, shadow_csm.wgsl, ddgi.wgsl)
- GAPSET_1_CORE: Type system, component store, math library fundamentals
- Frame graph IR: `crates/renderer-backend/src/frame_graph/mod.rs` (108KB)
