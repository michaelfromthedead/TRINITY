# PHASE 3 TODO: Shader Implementation

## Objective

Implement WGSL shaders for all post-processing effects, using Python implementations as authoritative reference.

---

## T-PP-3.1: Common Shader Library

**File**: `shaders/postprocess/common.wgsl`

### Tasks

- [ ] **T-PP-3.1.1**: Color space conversions
  - `srgb_to_linear(c: vec3f) -> vec3f`
  - `linear_to_srgb(c: vec3f) -> vec3f`
  - `rgb_to_luminance(c: vec3f) -> f32` (Rec. 709 coefficients)

- [ ] **T-PP-3.1.2**: Matrix utilities
  - `mat3x3_mul_vec3(m: mat3x3f, v: vec3f) -> vec3f`
  - ACES sRGB-to-ACEScg matrix constant

- [ ] **T-PP-3.1.3**: Sampling utilities
  - `sample_bilinear(tex, sampler, uv) -> vec4f`
  - `sample_bicubic(tex, sampler, uv) -> vec4f`

- [ ] **T-PP-3.1.4**: Math utilities
  - `saturate(x: f32) -> f32`
  - `remap(x, in_lo, in_hi, out_lo, out_hi) -> f32`

### Acceptance Criteria

- All utility functions tested in isolation
- Color space conversions match Python implementation

---

## T-PP-3.2: Tonemapping Shaders

**File**: `shaders/postprocess/tonemapping.wgsl`

### Tasks

- [ ] **T-PP-3.2.1**: Reinhard operator
  ```wgsl
  fn tonemap_reinhard(hdr: vec3f) -> vec3f {
      return hdr / (vec3f(1.0) + hdr);
  }
  ```

- [ ] **T-PP-3.2.2**: Reinhard Extended operator
  - Include white point parameter
  - Reference: ReinhardExtended.tonemap_value()

- [ ] **T-PP-3.2.3**: ACES operator
  - sRGB to ACEScg matrix multiply
  - RRT + ODT approximation
  - Reference: ACES.tonemap_value()

- [ ] **T-PP-3.2.4**: ACES Fitted operator
  - Krzysztof Narkowicz approximation
  - Formula: `(x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510) + 0.238081)`
  - Reference: ACESFitted._rrt_odt()

- [ ] **T-PP-3.2.5**: AgX operator
  - Log encoding
  - Look transform
  - Reference: AgX.tonemap_value()

- [ ] **T-PP-3.2.6**: Filmic operator
  - Shoulder and toe curves
  - John Hable formula
  - Reference: Filmic.tonemap_value()

- [ ] **T-PP-3.2.7**: Custom Curve operator
  - Hermite interpolation
  - Uniform buffer with control points
  - Reference: CustomCurve._hermite_interpolate()

- [ ] **T-PP-3.2.8**: Main compute entry point
  - Select operator via uniform
  - Dispatch 8x8 thread groups

### Acceptance Criteria

- All 8 operators implemented
- Output matches Python reference within epsilon
- sRGB output (gamma applied)

---

## T-PP-3.3: Bloom Shaders

**File**: `shaders/postprocess/bloom.wgsl`

### Tasks

- [ ] **T-PP-3.3.1**: Threshold pass
  - Soft-knee threshold formula
  - Reference: BloomThreshold.apply()

- [ ] **T-PP-3.3.2**: Downsample pass (13-tap)
  - 13-tap filter for high quality
  - Handles mip generation

- [ ] **T-PP-3.3.3**: Horizontal Gaussian blur
  - Separable Gaussian kernel
  - Reference: _gaussian_blur() horizontal pass

- [ ] **T-PP-3.3.4**: Vertical Gaussian blur
  - Separable Gaussian kernel
  - Reference: _gaussian_blur() vertical pass

- [ ] **T-PP-3.3.5**: Kawase blur (alternative)
  - 5-point cross pattern
  - Reference: _kawase_blur()

- [ ] **T-PP-3.3.6**: Upsample pass
  - Bilinear upsample
  - Additive blend with accumulator

- [ ] **T-PP-3.3.7**: Final composite
  - Add bloom to scene color
  - Apply intensity

### Acceptance Criteria

- Mip chain generated correctly
- Blur is smooth without banding
- Energy conserved (bright areas bloom, dark do not)

---

## T-PP-3.4: Color Grading Shaders

**File**: `shaders/postprocess/color_grading.wgsl`

### Tasks

- [ ] **T-PP-3.4.1**: White balance
  - Temperature to RGB matrix
  - Tint adjustment
  - Reference: WhiteBalance.apply()

- [ ] **T-PP-3.4.2**: Lift/Gamma/Gain
  - Per-channel adjustment
  - Reference: LiftGammaGain.apply()

- [ ] **T-PP-3.4.3**: Saturation
  - With vibrance (protect saturated colors)
  - Reference: SaturationSettings.apply()

- [ ] **T-PP-3.4.4**: LUT3D sampling
  - Trilinear interpolation from 3D texture
  - Reference: LUT3D.sample()

- [ ] **T-PP-3.4.5**: Channel mixer
  - 3x3 matrix multiply
  - Reference: ChannelMixer.apply()

- [ ] **T-PP-3.4.6**: Shadow/Midtone/Highlight separation
  - Blend weights for each range
  - Apply separate adjustments

- [ ] **T-PP-3.4.7**: Combined color grading pass
  - Apply all transforms in correct order
  - Single dispatch

### Acceptance Criteria

- LUT sampling matches reference
- All transforms chain correctly
- No color clipping

---

## T-PP-3.5: TAA Shaders

**File**: `shaders/postprocess/antialiasing.wgsl`

### Tasks

- [ ] **T-PP-3.5.1**: Motion vector sampling
  - Sample velocity buffer at reprojected location
  - Handle edge cases (off-screen)

- [ ] **T-PP-3.5.2**: History reprojection
  - Apply motion vector to find history UV
  - Bilinear sample history

- [ ] **T-PP-3.5.3**: Neighborhood clamping (AABB)
  - Build min/max AABB from 3x3 neighborhood
  - Clamp history to AABB

- [ ] **T-PP-3.5.4**: Temporal blend
  - Lerp between current and clamped history
  - Typical blend factor: 0.9 history, 0.1 current

- [ ] **T-PP-3.5.5**: FXAA (edge detection + blend)
  - Luminance edge detection
  - Subpixel anti-aliasing

### Acceptance Criteria

- Ghosting minimized
- Edges stable across frames
- No excessive blurring

---

## T-PP-3.6: Ambient Occlusion Shaders

**File**: `shaders/postprocess/ambient_occlusion.wgsl`

### Tasks

- [ ] **T-PP-3.6.1**: SSAO main pass
  - Hemisphere kernel sampling
  - Reference: SSAO.calculate()

- [ ] **T-PP-3.6.2**: HBAO main pass
  - Horizon-based angle sampling
  - Reference: HBAO.calculate()

- [ ] **T-PP-3.6.3**: GTAO main pass
  - Ground truth AO algorithm
  - Reference: GTAO.calculate()

- [ ] **T-PP-3.6.4**: Bilateral blur horizontal
  - Depth-aware blur
  - Reference: BilateralFilter

- [ ] **T-PP-3.6.5**: Bilateral blur vertical
  - Depth-aware blur

- [ ] **T-PP-3.6.6**: AO composite
  - Multiply AO into final image
  - Apply intensity

### Acceptance Criteria

- No visible halo artifacts
- AO respects depth discontinuities
- Performance acceptable at target resolution

---

## T-PP-3.7: DOF Shaders

**File**: `shaders/postprocess/dof.wgsl`

### Tasks

- [ ] **T-PP-3.7.1**: Calculate CoC
  - From depth buffer
  - Reference: CircleOfConfusion.calculate()

- [ ] **T-PP-3.7.2**: Near field blur
  - Blur foreground objects
  - Reference: NearFieldDOF.blur()

- [ ] **T-PP-3.7.3**: Far field blur
  - Blur background objects
  - Reference: FarFieldDOF.blur()

- [ ] **T-PP-3.7.4**: Bokeh kernel application
  - Apply bokeh shape (disk, polygon)
  - Reference: BokehShape.generate_kernel()

- [ ] **T-PP-3.7.5**: DOF composite
  - Blend near, far, and in-focus layers

### Acceptance Criteria

- Blur strength varies with CoC
- Bokeh shape visible in bright areas
- No artifacts at depth discontinuities

---

## T-PP-3.8: Exposure Shaders

**File**: `shaders/postprocess/exposure.wgsl`

### Tasks

- [ ] **T-PP-3.8.1**: Histogram clear
  - Zero out histogram buffer

- [ ] **T-PP-3.8.2**: Histogram bin
  - Per-pixel luminance binning
  - Atomic increment

- [ ] **T-PP-3.8.3**: Histogram reduce
  - Calculate weighted average
  - Find percentile value

- [ ] **T-PP-3.8.4**: Eye adaptation
  - Temporal smoothing
  - Asymmetric speeds
  - Reference: EyeAdaptation.update()

- [ ] **T-PP-3.8.5**: Apply exposure
  - Multiply color by exposure value

### Acceptance Criteria

- Histogram generated correctly
- Exposure adapts smoothly
- No popping on scene changes

---

## T-PP-3.9: Motion Blur Shaders

**File**: `shaders/postprocess/motion_blur.wgsl`

### Tasks

- [ ] **T-PP-3.9.1**: Tile max velocity
  - Reduce to per-tile max
  - Reference: TileMaxVelocity

- [ ] **T-PP-3.9.2**: Neighbor max
  - Expand max to neighboring tiles
  - For early-out optimization

- [ ] **T-PP-3.9.3**: Motion blur gather
  - Sample along velocity direction
  - Variable sample count based on velocity

### Acceptance Criteria

- Blur follows motion direction
- No visible tile boundaries
- Performance acceptable

---

## T-PP-3.10: Upscaling Shaders

**File**: `shaders/postprocess/upscaling.wgsl`

### Tasks

- [ ] **T-PP-3.10.1**: Bilinear upscale
  - Simple bilinear interpolation
  - Fallback quality

- [ ] **T-PP-3.10.2**: FSR1 EASU
  - Edge-adaptive spatial upsampling
  - AMD algorithm

- [ ] **T-PP-3.10.3**: FSR1 RCAS
  - Robust contrast-adaptive sharpening
  - Post-upscale sharpening

### Acceptance Criteria

- FSR1 quality better than bilinear
- Edges preserved
- Minimal ringing artifacts

---

## T-PP-3.11: Test Harness

### Tasks

- [ ] **T-PP-3.11.1**: GPU vs CPU comparison framework
  - Render with GPU shader
  - Render with CPU reference
  - Compute difference

- [ ] **T-PP-3.11.2**: Visual regression tests
  - Golden image comparison
  - Threshold for acceptable difference

- [ ] **T-PP-3.11.3**: Performance benchmarks
  - Measure each effect in isolation
  - Measure full post-process stack

### Acceptance Criteria

- All shaders match CPU reference
- Performance targets met
- No visual regressions

---

## Summary

| Task Group | Tasks | Priority |
|------------|-------|----------|
| T-PP-3.1 Common Library | 4 | Critical |
| T-PP-3.2 Tonemapping | 8 | High |
| T-PP-3.3 Bloom | 7 | High |
| T-PP-3.4 Color Grading | 7 | High |
| T-PP-3.5 TAA | 5 | High |
| T-PP-3.6 Ambient Occlusion | 6 | Medium |
| T-PP-3.7 DOF | 5 | Medium |
| T-PP-3.8 Exposure | 5 | Medium |
| T-PP-3.9 Motion Blur | 3 | Medium |
| T-PP-3.10 Upscaling | 3 | Medium |
| T-PP-3.11 Test Harness | 3 | High |

**Total Tasks**: 56
