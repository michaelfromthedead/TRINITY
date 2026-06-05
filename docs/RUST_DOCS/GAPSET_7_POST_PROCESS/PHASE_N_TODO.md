# GAPSET_7_POST_PROCESS: Phase TODO Tasks

**TASK_ID format**: T-PP-{PHASE}.{N}
- T = Task
- PP = Post-Processing subsystem
- {PHASE} = Phase number (1-6)
- {N} = Sequential task number within phase

**Total tasks**: 29
**Total effects covered**: 17 + orchestrator

---

## Phase 1: HDR Pipeline & Core Tonemapping (7 tasks)

Gaps: S8-G12 (orchestrator), S8-G14 (HDR pipeline), S8-G2 (tonemapping), S8-G16 (auto-exposure), S8-G17 (eye adaptation), S8-G9 (vignette)

### T-PP-1.1 -- Post-Process Stack Orchestrator

- **Description**: Implement the `PostProcessStack` executor that chains effects in canonical order, handles conditional execution, and integrates with S1 Frame Graph.
- **Files**: `engine/rendering/postprocess/__init__.py` (enum), `engine/rendering/postprocess/postprocess_stack.py`
- **Acceptance criteria**:
  - `PostProcessEffect` IntEnum with all 13 positions (0-120, sparse)
  - `PostProcessStack` class with sorted effect list
  - Each effect registered as a frame graph pass node
  - Conditional execution (disabled effects skipped per quality preset)
  - Barrier insertion between effects on aliased memory
  - 4 quality preset definitions that disable/enable effects
- **Dependencies**: S1 Frame Graph (frame graph pass registration), S14 RHI (command lists, bind groups)
- **Effort**: Medium

### T-PP-1.2 -- HDR Framebuffer Resource Management

- **Description**: Declare and manage the HDR scene target (RGBA16F), transient per-effect targets, and LDR output (RGBA8_UNORM) through the frame graph resource manager.
- **Files**: `engine/rendering/postprocess/postprocess_stack.py`
- **Acceptance criteria**:
  - HDR scene target registered as RGBA16F transient
  - LDR output target registered as RGBA8_UNORM
  - Transient buffer aliasing groups defined per Phase Architecture
  - Resource lifetime analysis matches non-overlapping effect windows
  - Peak VRAM reduction 30-50% vs. per-effect independent allocation
- **Dependencies**: S1 Frame Graph (resource aliasing), T-PP-1.1
- **Effort**: Medium

### T-PP-1.3 -- ACES Filmic Tone Mapping Shader

- **Description**: Implement the ACES Filmic tone mapping compute shader (Narkowicz 2015 5-parameter rational curve) and its Python config binding.
- **Files**: `engine/rendering/postprocess/tonemapping.py`, `shaders/tonemap_aces.comp.wgsl`
- **Acceptance criteria**:
  - ACES curve: `(x*(a*x+b))/(x*(c*x+d)+e)` with a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
  - Output clamped to [0, 1] range
  - Exposure multiplier applied before curve: `tonemap(exposure * hdr_color)`
  - Python config class with operator select, white point, exposure multiplier
  - Single compute dispatch, full-screen thread group
  - 16x16 thread groups, RGBA16F input -> RGBA8_UNORM output
- **Dependencies**: T-PP-1.1, S14 RHI (compute pipeline creation)
- **Effort**: Small

### T-PP-1.4 -- AgX, Reinhard, Uncharted 2 Tone Mapping Shaders

- **Description**: Implement the remaining 3 tone mapping operators: AgX (Ultra), Reinhard (Low), Uncharted 2 (legacy).
- **Files**: `engine/rendering/postprocess/tonemapping.py`, `shaders/tonemap_agx.comp.wgsl`, `shaders/tonemap_reinhard.comp.wgsl`, `shaders/tonemap_uncharted2.comp.wgsl`
- **Acceptance criteria**:
  - AgX: log-encoded color space with gamut mapping, ~15 ops
  - Reinhard: luminance-preserving `x/(x+1.0)` with `l_mapped/l` ratio
  - Uncharted 2: Hable filmic curve with white point adjustment
  - All 4 operators selectable via quality preset
  - Each operator validates against reference images
- **Dependencies**: T-PP-1.3
- **Effort**: Medium

### T-PP-1.5 -- Auto-Exposure (Histogram + Eye Adaptation)

- **Description**: Implement the full auto-exposure pipeline: luminance histogram compute, average luminance extraction, target exposure calculation, and exponential-moving-average eye adaptation.
- **Files**: `engine/rendering/postprocess/exposure.py`, `shaders/exposure_histogram.comp.wgsl`
- **Acceptance criteria**:
  - Log2 luminance histogram with configurable bin count (64/128/256/512)
  - 16x16 thread groups, atomic counters on R32_UINT buffer
  - Percentile culling (top/bottom N%) to exclude outliers
  - Target exposure: `middle_gray(0.18) / avg_luminance`, clamped [1/64, 64]
  - Eye adaptation EMA with asymmetric speed (dark-to-bright: 0.1s, bright-to-dark: 0.3s)
  - Exposure value stored in uniform buffer read by tonemapping
  - 4 quality levels with bin count and cull rate variation
- **Dependencies**: T-PP-1.1
- **Effort**: Medium

### T-PP-1.6 -- Vignette Shader

- **Description**: Implement the vignette post-process effect (radial darkening).
- **Files**: `engine/rendering/postprocess/cosmetic.py`, `shaders/vignette.comp.wgsl`
- **Acceptance criteria**:
  - Aspect-ratio-corrected radial distance from center
  - Configurable inner/outer radius, strength, feather exponent
  - Configurable vignette color (default black)
  - Smoothstep falloff function
  - Single compute dispatch, negligible cost (<0.005ms)
- **Dependencies**: T-PP-1.1
- **Effort**: Small

### T-PP-1.7 -- Phase 1 Integration Tests

- **Description**: End-to-end tests for the Phase 1 pipeline: HDR -> exposure -> tonemapping -> vignette -> LDR.
- **Files**: `tests/test_postprocess_phase1.py`
- **Acceptance criteria**:
  - Auto-exposure histogram matches CPU reference computation
  - Adaptation converges to target exposure within tolerance
  - All 4 tone mapping operators produce correct sRGB output vs. reference images
  - Vignette produces correct radial falloff
  - Full stack produces visible output on screen
  - Quality presets (Low/Medium/High/Ultra) each meet performance budget
- **Dependencies**: T-PP-1.1 through T-PP-1.6
- **Effort**: Medium

---

## Phase 2: Bloom & Lens Effects (5 tasks)

Gaps: S8-G1 (bloom), S8-G11 (lens flare), S8-G10 (chromatic aberration)

### T-PP-2.1 -- Bloom Bright-Pass and Downsample Pyramid

- **Description**: Implement bloom bright-pass extraction and the multi-level downsample chain (5-6 levels).
- **Files**: `engine/rendering/postprocess/bloom.py`, `shaders/bloom_brightpass.comp.wgsl`, `shaders/bloom_downsample.comp.wgsl`
- **Acceptance criteria**:
  - Bright-pass: `max(rgb - threshold, 0.0)` with soft knee option
  - 4x4 tent filter downsample using bilinear hardware taps at half-pixel offsets
  - 5 levels (full -> 1/2 -> 1/4 -> 1/8 -> 1/16 -> 1/32), 6 on Ultra
  - Each downsample dispatch is half-resolution of previous
  - Separable downsample passes for 5x5 Gaussian on High/Ultra
  - Async compute eligible pipelines
- **Dependencies**: T-PP-1.1
- **Effort**: Medium

### T-PP-2.2 -- Bloom Upsample-and-Blur and Composite

- **Description**: Implement the bottom-up upsample with blur and final composite with the HDR scene.
- **Files**: `engine/rendering/postprocess/bloom.py`, `shaders/bloom_upsample.comp.wgsl`, `shaders/bloom_composite.comp.wgsl`
- **Acceptance criteria**:
  - 3x3 tent filter upsample (separable H+V passes)
  - 5x5 Gaussian upsample on High/Ultra
  - Each level blends with corresponding same-level downsample
  - Final composite: `hdr_scene + bloom_intensity * upsampled_bloom`
  - Bloom intensity range 0.0-2.0, configurable
  - Total bloom end-to-end within ~0.32ms @ 1080p budget
- **Dependencies**: T-PP-2.1
- **Effort**: Medium

### T-PP-2.3 -- Lens Flare (Ghosts, Halo, Streaks)

- **Description**: Implement lens flare using bloom bright-pass output for ghost generation, halo, and anamorphic streaks.
- **Files**: `engine/rendering/postprocess/lens_flare.py`, `shaders/lens_flare.comp.wgsl`
- **Acceptance criteria**:
  - Ghost generation: N ghosts mirrored across lens center, progressive chromatic shift
  - Halo: central glow from brightest region, cubic falloff
  - Anamorphic streaks: horizontal direction with configurable spacing and falloff
  - Reuses bloom bright-pass texture as input (no redundant extraction)
  - Quality levels: Off/3 ghosts/6 ghosts+halo/8 ghosts+halo+streaks
  - Composite: `scene_color + lens_flare * flare_intensity`
  - Budget: ~0.08ms total
- **Dependencies**: T-PP-2.1 (bright-pass reuse)
- **Effort**: Medium

### T-PP-2.4 -- Chromatic Aberration Shader

- **Description**: Implement chromatic aberration with RGB radial offset, anamorphic distortion, and fringe suppression.
- **Files**: `engine/rendering/postprocess/cosmetic.py`, `shaders/chromatic_aberration.comp.wgsl`
- **Acceptance criteria**:
  - Per-channel radial offset: red outward, green reference, blue inward
  - Anamorphic distortion ratio (default 1.0, anamorphic ~1.33)
  - Fringe suppression near center (smoothstep 0.0-0.1) and extreme edges (smoothstep 0.45-0.5)
  - Quality levels: Off / 2px / 5px / 10px max offset
  - Operates in HDR space (before tone mapping)
  - Budget: ~0.013ms
- **Dependencies**: T-PP-1.1
- **Effort**: Small

### T-PP-2.5 -- Phase 2 Integration Tests

- **Description**: End-to-end tests for bloom, lens flare, and chromatic aberration.
- **Files**: `tests/test_postprocess_phase2.py`
- **Acceptance criteria**:
  - Bright-pass extraction preserves energy (sum of output matches sum of bright regions)
  - Down/upsample chain produces correct glow with no energy loss
  - Lens flare ghost positions match reference rendering
  - Chromatic aberration RGB offsets correct at known pixel positions
  - All effects produce expected results across 4 quality presets
- **Dependencies**: T-PP-2.1 through T-PP-2.4, T-PP-1.7
- **Effort**: Medium

---

## Phase 3: Cinematic Effects (6 tasks)

Gaps: S8-G3 (DOF), S8-G4 (motion blur), S8-G8 (film grain)

### T-PP-3.1 -- DOF Circle of Confusion Calculation

- **Description**: Implement the CoC calculation shader and near/far field separation.
- **Files**: `engine/rendering/postprocess/dof.py`, `shaders/dof_coc.comp.wgsl`
- **Acceptance criteria**:
  - CoC formula: `|f^2 * (depth - focal_dist) / (aperture * depth * (focal_dist - f))|`
  - Normalized to pixels, clamped to max_coc_radius * 2
  - CoC buffer: R16_FLOAT, sign indicates foreground/background
  - Configurable camera parameters: focal_distance, focal_length, aperture, sensor_size
  - Tile max CoC reduction (16x16 or 8x8 tiles per quality)
  - Quality levels control max CoC (4/8/16/32 pixels)
- **Dependencies**: T-PP-1.1
- **Effort**: Medium

### T-PP-3.2 -- DOF Bokeh Gather and Composite

- **Description**: Implement the tiled bokeh gather pass with configurable kernel shape and composite with in-focus regions.
- **Files**: `engine/rendering/postprocess/dof.py`, `shaders/dof_gather.comp.wgsl`, `shaders/dof_composite.comp.wgsl`
- **Acceptance criteria**:
  - Tile-based gather: each pixel gathers from neighbors within tile_max_coc radius
  - Bokeh shapes: circle (Low), hexagon (Medium/High), physical aperture Poisson disk (Ultra)
  - Adaptive kernel size per pixel based on per-pixel CoC
  - Foreground/background separation and composite: `lerp(scene, blurred, coc_abs/max_coc)`
  - Quality levels control sample count and bokeh shape
  - Total DOF budget: ~0.35-1.05ms
- **Dependencies**: T-PP-3.1
- **Effort**: Large

### T-PP-3.3 -- Motion Blur Velocity Tile Max and Sampling

- **Description**: Implement motion blur tile max reduction and per-pixel blur sampling with interleaved sampling.
- **Files**: `engine/rendering/postprocess/motion_blur.py`, `shaders/motionblur_tilemax.comp.wgsl`, `shaders/motionblur_sample.comp.wgsl`
- **Acceptance criteria**:
  - Tile max velocity reduction (16x16 tiles), optional dilation
  - Per-pixel blur: sample along velocity direction, center-weighted Gaussian
  - Rotated interleaved sampling (4x4 pattern) to reduce visible sample count
  - Configurable sample count (4/8/12/16 per quality)
  - Velocity buffer resolution: 1/4 (Low) / 1/2 (Medium/High) / full (Ultra)
  - Total motion blur budget: ~0.16-0.66ms
- **Dependencies**: T-PP-1.1, S2 GPU-Driven (velocity buffer)
- **Effort**: Medium

### T-PP-3.4 -- Motion Blur Bilateral Denoise

- **Description**: Implement the separable bilateral denoise pass for cleaning up interleaved motion blur sampling artifacts.
- **Files**: `engine/rendering/postprocess/motion_blur.py`, `shaders/motionblur_denoise.comp.wgsl`
- **Acceptance criteria**:
  - Separable bilateral blur: horizontal pass then vertical pass
  - Spatial weight: Gaussian falloff
  - Depth weight: exponential `exp(-|depth_diff| * falloff)`
  - Denoise radius configurable per quality level
  - Optional on Medium, mandatory on High/Ultra
  - Budget: ~0.05-0.15ms
- **Dependencies**: T-PP-3.3
- **Effort**: Small

### T-PP-3.5 -- Film Grain Shader

- **Description**: Implement procedural film grain with Gaussian shaping, luminance modulation, and optional chrominance grain.
- **Files**: `engine/rendering/postprocess/cosmetic.py`, `shaders/film_grain.comp.wgsl`
- **Acceptance criteria**:
  - Wang hash per pixel seeded by position + frame index
  - Gaussian-shaped grain via sum of uniform values
  - Luminance modulation: grain visible in midtones, faded in shadows/highlights
  - Optional chrominance grain in dark regions
  - Quality levels: Off / Uniform / Gaussian / Gaussian+chroma
  - Budget: ~0.008ms
- **Dependencies**: T-PP-1.1
- **Effort**: Small

### T-PP-3.6 -- Phase 3 Integration Tests

- **Description**: End-to-end tests for DOF, motion blur, and film grain.
- **Files**: `tests/test_postprocess_phase3.py`
- **Acceptance criteria**:
  - DOF CoC values match CPU reference at known depth
  - Bokeh gather produces correct blur radius vs. CoC size
  - Motion blur velocity direction matches screen-space movement
  - Film grain produces noise with correct luminance modulation curve
  - All effects work across 4 quality presets within budget
- **Dependencies**: T-PP-3.1 through T-PP-3.5
- **Effort**: Medium

---

## Phase 4: Ambient Occlusion (4 tasks)

Gaps: S8-G5 (SSAO/HBAO+/GTAO)

### T-PP-4.1 -- SSAO Compute + Bilateral Blur

- **Description**: Implement Screen-Space Ambient Occlusion with hemisphere sampling, noise rotation, and bilateral blur.
- **Files**: `engine/rendering/postprocess/ambient_occlusion.py`, `shaders/ssao.comp.wgsl`
- **Acceptance criteria**:
  - Hemisphere sample kernel (8-64 samples), pre-generated, uploaded as uniform buffer
  - 4x4 random rotation noise texture tiled across screen
  - Occlusion: project sample to screen space, compare depths
  - Range check to avoid false occlusion from distant geometry
  - Separable bilateral blur (depth + normal aware)
  - Quality levels: 8/16/32/64 samples
  - Output: R16_FLOAT occlusion factor
  - Budget: ~0.25-0.70ms
- **Dependencies**: T-PP-1.1
- **Effort**: Medium

### T-PP-4.2 -- HBAO+ Horizon Ray Marching

- **Description**: Implement Horizon-Based Ambient Occlusion+ with multi-step horizon ray marching.
- **Files**: `engine/rendering/postprocess/ambient_occlusion.py`, `shaders/hbao.comp.wgsl`
- **Acceptance criteria**:
  - Multi-step horizon ray marching (2-8 rays x 3-8 steps per quality)
  - Per-ray horizon angle search along tangent plane
  - Cosine-weighted angle integration
  - Falloff function: `1.0 - smoothstep(0.0, radius, distance)`
  - 20-30% faster than SSAO at equivalent quality
  - Quality levels: 2 rays x 3 steps / 4x4 / 6x6 / 8x8
  - Budget: ~0.26ms
- **Dependencies**: T-PP-4.1 (infrastructure, resource bindings)
- **Effort**: Medium

### T-PP-4.3 -- GTAO with Bent Normals and Multi-Bounce

- **Description**: Implement Ground-Truth Ambient Occlusion with cosine-weighted importance sampling, dual horizon search, multi-bounce approximation, and bent normal output.
- **Files**: `engine/rendering/postprocess/ambient_occlusion.py`, `shaders/gtao.comp.wgsl`
- **Acceptance criteria**:
  - Cosine-weighted hemisphere sampling (not random like SSAO)
  - Dual horizon search (upper + lower angles per direction)
  - Multi-bounce approximation: `(1+albedo)*single / (1+albedo*(1-single))`
  - Bent normal computation for directional diffuse occlusion
  - Optional temporal filter for frame stability
  - Quality levels: 3 rays/4 steps / 4x6 / 6x8 / 8x12
  - Budget: ~0.41ms
- **Dependencies**: T-PP-4.2 (infrastructure reuse)
- **Effort**: Large

### T-PP-4.4 -- Phase 4 Integration Tests

- **Description**: Tests for all three AO methods with correctness and performance validation.
- **Files**: `tests/test_postprocess_phase4.py`
- **Acceptance criteria**:
  - SSAO occlusion values in [0, 1] at every pixel
  - HBAO+ horizon angles match reference
  - GTAO bent normal points toward unoccluded direction
  - Multi-bounce GTAO produces lighter AO than single-bounce
  - All AO methods pass bilateral blur edge-preservation test
  - AO combine correctly blends into HDR scene at position 6 in chain
- **Dependencies**: T-PP-4.1 through T-PP-4.3
- **Effort**: Medium

---

## Phase 5: Color Grading (3 tasks)

Gaps: S8-G13 (color grading LUT)

### T-PP-5.1 -- 3D LUT Loader (.cube Parser + Atlas Baking)

- **Description**: Implement the .cube file parser and LUT-to-2D-atlas conversion for GPU upload.
- **Files**: `engine/rendering/postprocess/color_grading.py`
- **Acceptance criteria**:
  - Parse TITLE, LUT_3D_SIZE, DOMAIN_MIN/MAX, and data rows from .cube files
  - Convert N^3 entries to 2D atlas texture: width = N*N, height = N*ceil(N/2)
  - Support 16^3 through 64^3 LUT sizes
  - Atlas format: RGBA16F for GPU upload
  - Default LUT generation (identity grade) when no .cube file provided
  - LUT caching: reload on file change
- **Dependencies**: T-PP-1.1, S16 Asset Pipeline (file loading)
- **Effort**: Medium

### T-PP-5.2 -- 3D LUT Application Shader (Trilinear Interpolation)

- **Description**: Implement the LUT application compute shader using hardware-filtered trilinear sampling from the 2D atlas.
- **Files**: `engine/rendering/postprocess/color_grading.py`, `shaders/lut_apply.comp.wgsl`
- **Acceptance criteria**:
  - Single texture sample via hardware bilinear filtering on 2D atlas
  - LUT lookup in display-referred sRGB space (post-tonemap)
  - Support all 4 LUT sizes (16^3/32^3/48^3/64^3) via configurable atlas layout
  - Optional dither on Ultra (1/2 LSB noise)
  - Quality levels map to LUT size
  - Budget: ~0.02ms (constant regardless of LUT size)
- **Dependencies**: T-PP-5.1
- **Effort**: Small

### T-PP-5.3 -- Phase 5 Integration Tests

- **Description**: Tests for LUT loading and application.
- **Files**: `tests/test_postprocess_phase5.py`
- **Acceptance criteria**:
  - .cube parser handles all size variants (16^3/32^3/48^3/64^3)
  - Atlas layout produces correct hardware-filtered sampling
  - Identity LUT produces no color change (input == output)
  - Known LUT produces correct RGB output at grid points
  - LUT application produces smooth gradients (no banding)
- **Dependencies**: T-PP-5.1, T-PP-5.2
- **Effort**: Small

---

## Phase 6: Temporal AA & Upscaling (6 tasks)

Gaps: S8-G6 (TAA/TSR), S8-G7 (DLSS/FSR/XeSS)

### T-PP-6.1 -- TAA Halton Jitter and History Reprojection

- **Description**: Implement Halton sequence jitter injection into the projection matrix and history reprojection using the velocity buffer.
- **Files**: `engine/rendering/postprocess/antialiasing.py`, `shaders/taa_reproject.comp.wgsl`
- **Acceptance criteria**:
  - Halton(2,3) jitter sequence with configurable length (8/16/32/continuous)
  - Jitter applied as sub-pixel offset to projection matrix
  - History reprojection: `history_uv = current_uv - velocity`
  - Velocity buffer motion vector scaling correct for jitter offset
  - First-frame history initialization (no valid history)
- **Dependencies**: T-PP-1.1
- **Effort**: Medium

### T-PP-6.2 -- TAA Neighborhood Clamp and Blend

- **Description**: Implement neighborhood color clamping (AABB/variance clipping) and exponential moving average blend with disocclusion detection.
- **Files**: `engine/rendering/postprocess/antialiasing.py`, `shaders/taa_accumulate.comp.wgsl`
- **Acceptance criteria**:
  - 3x3 neighborhood AABB computation
  - Relaxed AABB (25% extension) on High, variance clipping (Salvi 2012) on Ultra
  - Optional YCoCg color space clipping (Ultra)
  - EMA blend: `lerp(clamped_history, current, blend_factor)`
  - Disocclusion detection via depth comparison, accelerated blend (up to 0.5)
  - Quality levels: box clamp / AABB / relaxed AABB / variance clip
  - Budget: ~0.06ms
- **Dependencies**: T-PP-6.1
- **Effort**: Medium

### T-PP-6.3 -- TSR Lanczos Upsampling

- **Description**: Implement Lanczos2/Lanczos3 upsampling for temporal super-resolution reconstruction.
- **Files**: `engine/rendering/postprocess/upscaling.py`, `shaders/tsr_upsample.comp.wgsl`
- **Acceptance criteria**:
  - Lanczos2 kernel (radius 2, 4x4 neighborhood) for Low/Medium/High
  - Lanczos3 kernel (radius 3, 6x6 neighborhood) for Ultra
  - Input scale: 50%/66%/75%/83% per quality
  - Output: full-resolution reconstructed frame
  - Exposure-weighted temporal accumulation for exposure change stability
  - Budget: ~0.10ms (Lanczos2) to ~0.15ms (Lanczos3)
- **Dependencies**: T-PP-6.2 (shares TAA infrastructure)
- **Effort**: Medium

### T-PP-6.4 -- TSR Adaptive Sharpening

- **Description**: Implement the adaptive sharpening filter applied after TSR temporal accumulation.
- **Files**: `engine/rendering/postprocess/upscaling.py`, `shaders/tsr_sharpen.comp.wgsl`
- **Acceptance criteria**:
  - Local contrast measurement from 3x3 box blur
  - Sharpening strength: `lerp(0.3, 0.8, contrast_measure)`
  - No overshoot on high-contrast edges
  - Quality levels control max sharpening strength
  - Budget: ~0.03ms
- **Dependencies**: T-PP-6.3
- **Effort**: Small

### T-PP-6.5 -- DLSS/FSR/XeSS Plugin Interface

- **Description**: Implement the abstract upscaler plugin interface and concrete implementations for DLSS (NGX), FSR (FidelityFX), and XeSS SDKs.
- **Files**: `engine/rendering/postprocess/upscaling.py`
- **Acceptance criteria**:
  - `UpscalerPlugin` abstract base class with `initialize()`, `evaluate()`, `get_optimal_render_resolution()`, `is_available`, `name`
  - `DLSSImplementation`: wraps NGX SDK initialization + per-frame evaluate
  - `FSRImplementation`: wraps FidelityFX FSR 2/3 context create + dispatch
  - `XeSSImplementation`: wraps XeSS context create + execute
  - Runtime auto-detection: DLSS -> XeSS -> FSR -> TSR fallback
  - Graceful degradation when SDK DLLs are not present
  - All upscalers share the same frame graph binding interface
- **Dependencies**: T-PP-6.4 (TSR is the fallback)
- **Effort**: Large

### T-PP-6.6 -- Phase 6 Integration Tests

- **Description**: Tests for TAA, TSR, and upscaler plugin.
- **Files**: `tests/test_postprocess_phase6.py`
- **Acceptance criteria**:
  - Halton sequence produces correct sub-pixel offsets
  - TAA converges to stable image over N frames (N = 1/blend_factor)
  - TAA disocclusion detection triggers correctly on depth changes
  - TSR Lanczos upsampling preserves edge sharpness vs. bilinear
  - TSR produces correct output at all input scales
  - Upscaler plugin interface creates correct frame graph pass nodes
  - Full post-process stack end-to-end: 17 effects in canonical order
  - All 4 quality presets meet performance budgets
- **Dependencies**: T-PP-6.1 through T-PP-6.5
- **Effort**: Large

---

## Task Summary

| Phase | Task ID Range | Effect Count | Total Tasks |
|-------|--------------|--------------|-------------|
| 1: HDR Pipeline & Core | T-PP-1.1 to T-PP-1.7 | 4 + orchestrator | 7 |
| 2: Bloom & Lens | T-PP-2.1 to T-PP-2.5 | 3 | 5 |
| 3: Cinematic | T-PP-3.1 to T-PP-3.6 | 3 | 6 |
| 4: Ambient Occlusion | T-PP-4.1 to T-PP-4.4 | 3 AO variants | 4 |
| 5: Color Grading | T-PP-5.1 to T-PP-5.3 | 1 | 3 |
| 6: Temporal & Upscaling | T-PP-6.1 to T-PP-6.6 | 4 | 6 |
| **Total** | **T-PP-1.1 to T-PP-6.6** | **17 effects + orchestrator** | **29 tasks** |
