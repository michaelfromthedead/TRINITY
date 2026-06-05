# PROJECT: engine/rendering/postprocess

## Scope

Post-processing subsystem comprising 10 source files (8,861 lines total) implementing a comprehensive post-processing framework with real CPU-side algorithms but stub GPU execution paths.

### Files In Scope

| File | Lines | Classification |
|------|-------|----------------|
| postprocess_stack.py | 1,776 | REAL |
| upscaling.py | 886 | PARTIAL |
| bloom.py | 840 | REAL |
| antialiasing.py | 733 | PARTIAL |
| tonemapping.py | 694 | REAL |
| color_grading.py | 681 | REAL |
| ambient_occlusion.py | 673 | PARTIAL |
| dof.py | 652 | PARTIAL |
| exposure.py | 590 | REAL |
| motion_blur.py | 550 | PARTIAL |

## Classification

**PARTIAL** — Real architecture with stub GPU execution.

The subsystem has production-quality CPU/Python implementations including:
- Correct mathematical formulas
- Proper effect chaining
- Quality presets
- Frame graph integration

However, actual GPU commands are not recorded. All `execute()` methods prepare data structures but return `None` or placeholder buffers.

## Goals

1. Verify and test all existing CPU-side mathematical implementations
2. Integrate GPU execution via RHI command list recording
3. Implement shader sources (WGSL/HLSL) for all effects
4. Connect stub methods to real GPU dispatch

## Constraints

- Must preserve existing API surface (PostProcessEffect ABC, settings dataclasses)
- Must maintain frame graph integration via `add_to_frame_graph()` and `PassNode`
- Must preserve quality preset system (Low/Medium/High/Ultra)
- Must maintain proper effect ordering by priority
- Must preserve temporal effect handling (`is_first_frame` logic)

## Dependencies

### Internal

- `engine.rendering.framegraph.frame_graph.FrameGraph`
- `engine.rendering.framegraph.pass_node.PassNode, PassFlags`
- `engine.rendering.framegraph.resource_manager.ResourceFormat`
- `./constants` module for magic numbers

### External

- RHI command list (not yet integrated)
- Shader compiler (WGSL/HLSL sources not present)

## What Actually Exists (REAL)

### PostProcessStack Framework
- Effect ordering by priority
- Quality preset system
- Execution flags (SKIP_IF_DISABLED, SKIP_ON_FIRST_FRAME, FORCE_ASYNC)
- Ping-pong intermediate target management
- PostProcessVolume with spatial blending (box/sphere shapes)
- Frame graph integration

### Tonemap Operators (8 operators)
- Reinhard, ReinhardExtended, ACES, ACESFitted, AgX, Filmic, CustomCurve
- ACES color space matrices (sRGB -> ACEScg)
- AgX log-space encoding with look transforms
- Filmic S-curve with shoulder/toe controls
- Custom curves with Hermite interpolation

### Bloom Pipeline
- Soft threshold with knee parameters
- Mip chain generation (up to 8 levels)
- Three blur algorithms: Gaussian (separable), Kawase (5-point cross), Box
- Per-mip intensity and scatter settings

### Color Grading
- White balance temperature/tint to RGB conversion
- Lift/Gamma/Gain per-channel adjustment
- Shadow/Midtone/Highlight separation with blend weights
- Saturation with vibrance
- 3D LUT loading (.cube format) with trilinear interpolation
- Channel mixer matrix

### Exposure Control
- Luminance-to-EV conversion (ISO 12232:2006)
- Center-weighted, spot, and matrix metering kernels
- Histogram-based percentile exposure
- Eye adaptation temporal smoothing with asymmetric speeds

### DOF Optics
- Circle of Confusion calculation from aperture/focal length/sensor size
- Hyperfocal distance computation
- Bokeh kernel generation (disk, polygon with blade curvature, anamorphic)
- AutoFocus with smooth transition

### TAA/AA (partial)
- Halton jitter sequence generation
- Projection matrix jitter application

### AO/Motion Blur (partial)
- SSAO hemisphere kernel generation
- HBAO direction generation
- Bilateral filter weight formula
- Tile-based velocity structure

## What Is STUB

All `execute()` methods that would record GPU commands return placeholder values:
- `_ao_buffer`, `_output_buffer`, `_motion_buffer` are `None`
- GPU shader dispatch is not implemented
- No RHI command list recording

## Acceptance Criteria

### Phase 1: CPU Algorithm Verification
- [ ] Unit tests for all tonemap operators with known input/output pairs
- [ ] Unit tests for bloom blur algorithms
- [ ] Unit tests for color grading transforms
- [ ] Unit tests for exposure calculations
- [ ] Unit tests for DOF optics (CoC, hyperfocal distance, bokeh kernels)
- [ ] Unit tests for Halton sequence generation
- [ ] Unit tests for SSAO kernel generation

### Phase 2: GPU Integration
- [ ] RHI command list integration in all `execute()` methods
- [ ] Proper GPU buffer allocation replacing `None` placeholders
- [ ] Intermediate target management connected to real GPU resources
- [ ] Frame graph passes execute actual GPU work

### Phase 3: Shader Implementation
- [ ] WGSL/HLSL shaders for all 8 tonemap operators
- [ ] WGSL/HLSL shaders for bloom pipeline (threshold, blur, upsample)
- [ ] WGSL/HLSL shaders for color grading (LUT sampling, channel mixing)
- [ ] WGSL/HLSL shaders for TAA (history reprojection, clamping, blend)
- [ ] WGSL/HLSL shaders for SSAO/HBAO/GTAO
- [ ] WGSL/HLSL shaders for DOF (CoC, blur, bokeh)
- [ ] WGSL/HLSL shaders for motion blur
- [ ] WGSL/HLSL shaders for upscaling (FSR1, bilinear)
