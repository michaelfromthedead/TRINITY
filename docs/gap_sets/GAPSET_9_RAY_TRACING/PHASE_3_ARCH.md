# PHASE_3_ARCH.md — Full Path Tracing and Neural Denoising

> **Phase**: 3 of 3
> **Status**: Future work
> **Tasks**: 6 (all [-])
> **Gaps Covered**: S10-G7 (neural denoising), S10-G1 (full API stability)
> **Prerequisites**: Phase 2 complete, wgpu RT pipeline stable, NPU/tensor core extensions
> **Reality (2026-05-22)**: Zero implementation. All 6 tasks are [-] not started.

---

## 1. Architecture Overview

Phase 3 extends the ray tracing system with multi-bounce path tracing, temporal accumulation, neural denoising, and ML-predictive adaptive quality.

```
                    Path Tracing Pipeline
  ┌───────────────────────────────────────────────────────┐
  │                                                       │
  │  Each Frame: 1 sample/pixel                          │
  │  Temporal Accumulation: converges over 256 frames     │
  │                                                       │
  │  ┌──────────────┐    ┌──────────────────┐            │
  │  │  Generate Ray  │───▶  Trace Path      │            │
  │  │  (pixel→scene) │    │  (up to 4       │            │
  │  └──────────────┘    │   bounces)        │            │
  │                       └────────┬─────────┘            │
  │                                │                      │
  │                       ┌────────▼─────────┐            │
  │                       │  Russian Roulette │            │
  │                       │  Termination     │            │
  │                       └────────┬─────────┘            │
  │                                │                      │
  │                       ┌────────▼─────────┐            │
  │                       │  Temporal        │            │
  │                       │  Accumulation    │            │
  │                       └────────┬─────────┘            │
  │                                │                      │
  │                       ┌────────▼─────────┐            │
  │                       │  Neural Denoiser │            │
  │                       │  (U-Net)         │            │
  │                       └──────────────────┘            │
  │                                                       │
  └───────────────────────────────────────────────────────┘
```

## 2. Component Details

### 2.1 Full Path Tracing Shader [-] (T-RT-P3.1)

**Required WGSL compute shader**:
```
Input:  G-Buffer (depth, normal, albedo)
        TLAS (scene geometry)
        Light list (emissive geometry + analytic lights)

Ray Generation:
  per pixel:
    - Generate primary ray from camera
    - Trace through TLAS
    - At hit: sample BSDF (BRDF + BTDF)

Path Bounces (up to 4):
    - Russian roulette: terminate with P = max(albedo * (1 - metallic), 0.2)
    - Next event estimation: sample direct light at each bounce
    - Accumulate direct + indirect illumination

Output: Per-pixel accumulated radiance (HDR)
```

### 2.2 Temporal Accumulation [-] (T-RT-P3.2)

**Algorithm**:
```
accumulated_pixel = lerp(prev_accumulated_pixel, current_sample, 1.0 / frame_count)
```

Key features:
- Per-pixel accumulation count buffer (u32).
- Camera cut detection via threshold on delta view-projection.
- Scene change detection via instance count/transform delta.
- Motion-based bias: clamp alpha to 0.05-0.2 on movement.
- Max 256 frames accumulation (reset on overflow).

### 2.3 Neural Denoising (U-Net) [-] (T-RT-P3.3)

**Architecture**:
```
Input (7 channels):
  RGB noisy + depth + normal + roughness + albedo

U-Net:
  Encoder 1: 3x3 conv, 32 channels
  Encoder 2: 3x3 conv, 64 channels (downsampled)
  Decoder 2: 3x3 deconv, 32 channels (upsampled + skip from Encoder 2)
  Decoder 1: 3x3 deconv, 16 channels (+ skip from Encoder 1)
  Output: 3x3 conv, 3 channels (denoised RGB)

Activations: ReLU between stages
Weights: Loaded from ONNX-derived storage buffer
Fallback: Spatial + Temporal denoiser (Phase 1 + Phase 2)
```

**Performance target**: < 3ms at 1080p.

### 2.4 Research Tasks [-] (P3.4, P3.5)

**T-RT-P3.4 (Denoising Survey)**:
Candidates to evaluate: U-Net, KPCN (Kernel-Predicting Convolutional Network), NFN (Neural-Filtering Network), SVGF (Spatiotemporal Variance-Guided Filtering), BMFR (Blockwise Multi-Order Feature Regression).

**T-RT-P3.5 (OMM/DMM Timeline)**:
Survey wgpu GitHub for Opacity Micromap and Displacement Micromap support. NVIDIA SER (Shader Execution Reordering) impact. Driver support matrix (NVIDIA, AMD, Intel).

### 2.5 Adaptive Quality 2.0 [-] (T-RT-P3.6)

**ML Prediction**:
```
Features: instance_count, ray_count, material_count, resolution, denoiser_iterations
Model: Linear regression or small NN (2-layer, 16 hidden)
Target: Predict frame time before rendering
Action: Proactively adjust quality if predicted time exceeds budget
Fallback: Reactive system (Phase 2, P2.13) when confidence < 0.7
```

## 3. File Map

| Task | New Files Required |
|------|-------------------|
| P3.1 | `shaders/rt_pathtrace.comp.wgsl` |
| P3.2 | `engine/rendering/rt/path_trace_accum.py` |
| P3.3 | `shaders/denoiser_neural.comp.wgsl`, `engine/rendering/rt/neural_denoiser.py` |
| P3.4 | `docs/research/denoising_survey.md` |
| P3.5 | `docs/research/rt_extensions_timeline.md` |
| P3.6 | `engine/rendering/rt/adaptive_quality_ml.py` |

## 4. Data Flow (Frame)

```
Frame Start
  │
  ├─ ML Prediction (P3.6)
  │   └─ Predict frame time from scene features
  │   └─ Proactively adjust quality settings
  │
  ├─ AS Build Phase (P1.2, P1.3)
  │
  ├─ Path Tracing (P3.1)
  │   ├─ 1 sample/pixel, up to 4 bounces
  │   ├─ Russian roulette termination
  │   └─ Output: HDR radiance buffer
  │
  ├─ Temporal Accumulation (P3.2)
  │   ├─ Accumulate over frames
  │   ├─ Camera/scene change detection
  │   └─ Output: accumulated buffer
  │
  ├─ Neural Denoiser (P3.3)
  │   ├─ U-Net inference
  │   └─ Output: denoised final buffer
  │
  └─ Tonemapping / Display
```

## 5. Cross-Phase Dependencies (Verified)

```
P2.4 (RT GI shaders) ─▶ P3.1 (Path tracing extends GI shaders)
P2.9 (Three-stage denoiser) ─▶ P3.3 (Neural denoiser fallback)
P2.13 (Adaptive quality) ─▶ P3.6 (Adds ML prediction)
P3.4 (Denoising survey) ─▶ P3.3 (Informs architecture choice)
```

All Phase 3 dependencies are correct and verified against the codebase.
