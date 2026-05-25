# GAPSET 7: Post-Processing Stack -- Project Overview

**GAPSET ID**: GAPSET_7_POST_PROCESS
**SCOPE**: 17 post-processing effects + stack orchestrator + frame graph integration
**SOURCE INVESTIGATION DATE**: 2026-05-22

---

## Project Structure

### Rust Backend (`crates/renderer-backend/src/`)

| File | Lines | Role |
|------|-------|------|
| `post_process.rs` | 480 | Frame graph IR pass builders for 3 effects + chain wiring + 22 unit tests |

The Rust side builds intermediate-representation pass nodes (`IrPass`, `IrResource`) for the frame graph. It does not execute any GPU work -- it constructs a DAG of passes and resources that the backend scheduler will later dispatch.

**Implemented passes:**
- `create_tonemap_pass()` -- ACES filmic tonemapping IR node (placeholder 1x1x1 dispatch)
- `create_bloom_pass()` -- Gaussian bloom IR node (dispatch derived from resolution: width/8 x height/8)
- `create_taa_pass()` -- Temporal AA IR node with history buffer read/write (placeholder 1x1x1 dispatch)
- `create_post_process_chain()` -- Wires all 3 passes: tonemap -> bloom -> TAA, with 3 transient RGBA16F resources

### Python Frontend (`engine/rendering/postprocess/`)

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 363 | Public symbol re-exports |
| `postprocess_stack.py` | 1,776 | Stack orchestrator, quality presets, effect ABC, executor, volume |
| `constants.py` | 423 | Centralized constants for all effect families |
| `tonemapping.py` | 694 | 8 tonemap operators (CPU reference) |
| `exposure.py` | 590 | 3 exposure modes + eye adaptation EMA (CPU) |
| `bloom.py` | 840 | Bloom pipeline: threshold, downsample, blur, upsample (CPU stubs) |
| `ambient_occlusion.py` | 673 | SSAO, HBAO, GTAO + bilateral filter (CPU stubs) |
| `motion_blur.py` | 550 | Camera, object, combined motion blur (CPU stubs) |
| `dof.py` | 652 | CoC, bokeh shapes, near/far field, auto focus (CPU stubs) |
| `color_grading.py` | 681 | .cube LUT parser, trilinear interp, LGG, white balance (CPU working) |
| `antialiasing.py` | 733 | TAA, FXAA, SMAA + Halton jitter (CPU stubs + working jitter) |
| `upscaling.py` | 886 | Spatial/temporal upscaler ABCs + DLSS/FSR/XeSS stubs |

### WGSL Shaders (`crates/renderer-backend/shaders/`)

**No post-process shaders exist.** The shaders/ directory contains only: pbr, shadow, shadow_csm, ddgi, light_culling, particles.

---

## Architecture

### Effect Pipeline Order

```
HDR Scene Color
    |
    v
[1] Exposure (Auto/Manual/Histogram + EyeAdaptation)
    |
    v
[2] Bloom (Bright-pass -> Downsample 5-level -> Blur -> Upsample -> Composite)
    |
    v
[3] Depth of Field (CoC -> Near/Far blur -> Bokeh -> Composite)
    |
    v
[4] Motion Blur (Tile max velocity -> Sampling -> Denoise)
    |
    v
[5] Ambient Occlusion (SSAO / HBAO / GTAO + Bilateral filter)
    |
    v
[6] Tonemapping (ACES / AgX / Reinhard / Filmic -> LDR)
    |
    v
[7] Color Grading (White balance -> LGG -> Contrast -> LUT)
    |
    v
[8] Anti-Aliasing (TAA + optional FXAA/SMAA)
    |
    v
[9] Upscaling (FSR / DLSS / XeSS / Bilinear)
    |
    v
LDR Output
```

### Dual-Path Problem

Both Rust and Python implement post-process concepts with different architectures:
- **Rust** builds frame graph IR nodes (pass planning, no execution)
- **Python** defines an OOP hierarchy with quality presets, volumes, and intermediate targets
- **Neither produces actual GPU work** -- Rust dispatches placeholder workgroups, Python execute() methods are empty stubs
- **No WGSL shaders** exist for any effect

### Quality Presets

| Level | Active Effects | Tonemap | Bloom | AA | Upscale |
|-------|---------------|---------|-------|----|---------|
| Low | 5 | Reinhard | Off | FXAA | Bilinear |
| Medium | 8 | ACES | 3-level | TAA | CAS |
| High | 10 | ACES | 5-level | TAA | FSR1 |
| Ultra | 13 | AgX | 7-level | TAA+SMAA | DLSS/XeSS |

---

## Dependencies

- **S1 Frame Graph**: Effect passes register as `IrPass` nodes with read/write resource sets
- **S14 RHI**: Resource handles, dispatch calls, barrier management
- **S2 GPU-Driven**: Velocity buffer (motion blur), depth buffer (AO, DOF), GBuffer (normal for AO)
- **S16 Asset Pipeline**: .cube LUT file loading

---

## Key Gaps

1. **Zero WGSL/HLSL shaders** -- all effects lack GPU implementation
2. **4 missing effects** -- vignette, chromatic aberration, lens flare, film grain have no code
3. **Python execute() stubs** -- all effect execute() methods return without pixel work
4. **Zero integration tests** -- 6 planned test phases have no coverage
5. **Rust/Python duplication** -- two different architectures for the same effects
6. **Constants bug** -- `HISTOGRAM_BINS_MAX=256` conflicts with Ultra preset requiring 512
