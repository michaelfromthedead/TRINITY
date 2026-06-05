# GAPSET 7: Post-Processing Stack -- Reality Summary

**GAPSET ID**: GAPSET_7_POST_PROCESS  
**SCOPE**: 17 effects + orchestrator, across Rust and Python codebases  
**PLANNED TASKS**: 70  
**SOURCE INVESTIGATION DATE**: 2026-05-22

---

## Files on Disk

### Rust (1 file, 480 lines)
| File | Lines | Reality |
|------|-------|---------|
| `/crates/renderer-backend/src/post_process.rs` | 480 | Has **3 pass builders** (ACES tonemap, Gaussian bloom, TAA) + chain builder + 22 tests |

### Python (12 files, ~9,341 lines)
| File | Lines | Reality |
|------|-------|---------|
| `postprocess_stack.py` | 1,776 | Stack orchestrator, quality presets, IntermediateTargetManager, PostProcessEffect ABC, PostProcessStack, PostProcessStackExecutor, PostProcessVolume |
| `constants.py` | 423 | Centralized constants for all 8 effect families |
| `tonemapping.py` | 694 | 8 operators (Reinhard, ACES full + fitted, AgX, Filmic/Hable, CustomCurve) |
| `exposure.py` | 590 | Manual, auto-average, histogram modes + EyeAdaptation EMA |
| `bloom.py` | 840 | Threshold, downsample pyramid, Gaussian/Kawase/Box blur, upsample |
| `ambient_occlusion.py` | 673 | SSAO, HBAO, GTAO with bilateral filter, bent normals |
| `motion_blur.py` | 550 | Camera, object, combined modes + TileMaxVelocity |
| `dof.py` | 652 | Circle of Confusion, near/far field, bokeh shapes, auto focus |
| `color_grading.py` | 681 | White balance, LGG, contrast, saturation, .cube LUT parser (trilinear interp) |
| `antialiasing.py` | 733 | FXAA, SMAA, TAA with Halton jitter, reprojection, neighborhood clamp |
| `upscaling.py` | 886 | Spatial (bilinear, FSR1, CAS) + Temporal (FSR2, DLSS, XeSS) |
| `__init__.py` | 363 | Re-exports all public symbols |

### WGSL Shaders
**No post-process shaders exist.** The `shaders/` directory contains only: pbr, shadow, shadow_csm, ddgi, light_culling, particles.

---

## Task Completion Reality

| Phase | Planned Tasks | [x] | [-] or [~] | Key Gaps |
|-------|--------------|-----|------------|----------|
| 1: HDR Pipeline & Core Tonemapping | 16 | 3 | 13 | No vignette, no auto-exposure histogram compute, no WGSL shaders |
| 2: Bloom & Lens Effects | 12 | 2 | 10 | No lens flare, no chromatic aberration, no WGSL shaders |
| 3: Cinematic Effects | 13 | 4 | 9 | No film grain, DOF/motion blur are CPU stubs, no integration tests |
| 4: Ambient Occlusion | 9 | 4 | 5 | SSAO/HBAO/GTAO are Python stubs (return None), no WGSL |
| 5: Color Grading | 7 | 4 | 3 | LUT engine works (.cube + trilinear), no 2D atlas conversion, no shader |
| 6: Temporal AA & Upscaling | 13 | 5 | 8 | TAA is CPU Python stub, no TSR, DLSS/FSR/XeSS are interface stubs |
| **Total** | **70** | **22** | **48** | |

---

## Critical Findings

### Finding 1: Rust/Python Duplication Without Convergence
The Rust `post_process.rs` and Python `engine/rendering/postprocess/` implement overlapping concepts with different architectures:
- Rust: IR pass builders (3 passes: tonemap -> bloom -> TAA), 480 lines, 22 tests
- Python: Full OOP hierarchy with quality presets, volumes, intermediate targets, 10 module, 9,341 lines
- **Neither produces actual GPU work** -- Rust builds frame graph IR nodes with placeholder dispatch (1x1x1 groups), Python execute() methods are empty stubs

### Finding 2: 4 Effects Have Zero Implementation
Four effects listed in the gap set plan have **no module at all** in either codebase:
- **Vignette** -- no Python module, no Rust function, not referenced in postprocess code
- **Chromatic Aberration** -- no module, not referenced
- **Lens Flare** -- no module, not referenced
- **Film Grain** -- no module, not referenced

### Finding 3: No WGSL/HLSL Shaders for Post-Processing
Zero shader files exist for any post-process effect. The `shaders/` directory contains only pbr, shadow, DDGI, light culling, and particles. Both Rust and Python code dispatch compute shaders but no actual shader source exists.

### Finding 4: Zero Integration Tests
The TODO calls for 6 phases of integration tests (T-PP-1.7, T-PP-2.5, T-PP-3.6, T-PP-4.4, T-PP-5.3, T-PP-6.6). Zero integration tests exist. Only the Rust `post_process.rs` has unit tests (22 tests for pass wiring).

### Finding 5: Python Effects Are Structural Stubs
Almost all Python effect `execute()` methods are empty (they `return` without doing work or setting output data). They define the architecture, settings classes, operators, and pipeline structure but do not perform actual pixel processing:
- `TonemappingEffect.execute()` -- empty
- `BloomEffect.execute()` -- loops over mips calling downsample/upsample/blur but all do nothing (CPU stubs)
- `AOEffect.execute()` -- calls calculate() methods that return None
- `MotionBlurEffect.execute()` -- calls apply_blur() that returns input unchanged
- `DOFEffect.execute()` -- calls blur() methods that return None
- `ColorGradingEffect.execute()` -- empty
- `AAEffect.execute()` -- calls apply() methods that return None
- `UpscalingEffect.execute()` -- calls upscale() methods that return None

### Finding 6: Quality Presets Don't Match Code
Python quality presets reference effects (DOF, MotionBlur, AmbientOcclusion, TAA, SMAA, FXAA) whose execute() methods are empty. The Rust side only has 3 passes (ACES tonemap, Gaussian bloom, TAA). No code exists for SMAA, FXAA, DOF, motion blur, or AO that produces GPU work.

---

## File Count Summary

| Category | Count |
|----------|-------|
| Rust source files | 1 (480 lines) |
| Python modules | 12 (~9,341 lines) |
| WGSL shaders | 0 |
| Integration tests | 0 |
| Unit tests (Rust only) | 22 |
| Document files (this gap set) | 0 (pre-existing) |

---

## Architecture Overview

```
HDR Scene Color
    |
    v
[Exposure] (Python stub, no compute)
    |
    v
[Bloom] (Python: architecture + CPU mip stubs / Rust: IR pass builder)
    |
    v
[Depth of Field] (Python stub, no compute)
    |
    v
[Motion Blur] (Python stub, no compute)
    |
    v
[Ambient Occlusion] (Python stub, no compute)
    |
    v
[Tonemapping] (Python: 8 operators with CPU impl / Rust: IR pass builder only)
    |
    v
[Color Grading] (Python: .cube LUT + CPU trilinear working / Rust: not present)
    |
    v
[Anti-Aliasing] (Python: TAA Halton jitter + reprojection stubs / Rust: IR pass builder)
    |
    v
[Upscaling] (Python: interface stubs for FSR/DLSS/XeSS / Rust: not present)
    |
    v
LDR Output
```

## Recommendations

1. Write actual WGSL compute shaders for all effects before attempting GPU integration
2. Choose one authoritative path (Rust frame graph IR or Python direct execution) -- dual paths add complexity without benefit
3. Implement the 4 missing modules (vignette, CA, lens flare, film grain) or remove them from scope
4. Build integration tests that verify pixel output, not just pass wiring
5. Replace the Python `execute()` stubs with actual CPU reference implementations for verification
