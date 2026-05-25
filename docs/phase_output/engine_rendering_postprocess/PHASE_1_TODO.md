# PHASE 1 TODO: CPU Algorithm Verification

## Objective

Create unit tests for all CPU-side mathematical implementations in the post-processing subsystem.

---

## T-PP-1.1: Tonemap Operator Tests

**File**: `tests/rendering/postprocess/test_tonemapping.py`

### Tasks

- [ ] **T-PP-1.1.1**: Test Reinhard.tonemap_value()
  - Input: 1.0 -> Expected: 0.5
  - Input: 4.0 -> Expected: 0.8
  - Input: 0.0 -> Expected: 0.0
  - Verify formula: `L / (1 + L)`

- [ ] **T-PP-1.1.2**: Test ReinhardExtended.tonemap_value()
  - Verify white point scaling
  - Test white_point=1.0 equals standard Reinhard

- [ ] **T-PP-1.1.3**: Test ACES.tonemap_value()
  - Verify sRGB -> ACEScg matrix multiplication
  - Test against official ACES reference values

- [ ] **T-PP-1.1.4**: Test ACESFitted._rrt_odt()
  - Verify approximation formula: `(x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510) + 0.238081)`
  - Test boundary: x=0 -> output=0

- [ ] **T-PP-1.1.5**: Test AgX.tonemap_value()
  - Verify log encoding
  - Verify look transform application

- [ ] **T-PP-1.1.6**: Test Filmic.tonemap_value()
  - Verify shoulder curve
  - Verify toe curve
  - Test linear section

- [ ] **T-PP-1.1.7**: Test CustomCurve._hermite_interpolate()
  - Verify smooth interpolation between control points
  - Verify endpoint matching

### Acceptance Criteria

- All 8 tonemap operators have tests
- Known-answer tests use published or hand-calculated values
- Edge cases: zero input, very large input, negative input handling

---

## T-PP-1.2: Bloom Algorithm Tests

**File**: `tests/rendering/postprocess/test_bloom.py`

### Tasks

- [ ] **T-PP-1.2.1**: Test BloomThreshold.apply()
  - Verify soft-knee threshold curve
  - Input below threshold -> near zero output
  - Input above threshold -> scaled output

- [ ] **T-PP-1.2.2**: Test BloomDownsample.generate_mip_chain()
  - Verify mip level count (up to 8)
  - Verify each level is half the previous

- [ ] **T-PP-1.2.3**: Test _gaussian_blur() energy conservation
  - Create synthetic image with known sum
  - Apply blur
  - Verify output sum equals input sum (within epsilon)

- [ ] **T-PP-1.2.4**: Test _gaussian_blur() kernel weights
  - Verify kernel weights sum to 1.0
  - Verify symmetric kernel

- [ ] **T-PP-1.2.5**: Test _kawase_blur() pattern
  - Verify 5-point cross sampling pattern
  - Verify offset calculation

- [ ] **T-PP-1.2.6**: Test _box_blur() uniformity
  - Verify uniform weights
  - Verify energy conservation

- [ ] **T-PP-1.2.7**: Test per-mip intensity settings
  - Verify scatter parameter application
  - Verify intensity scaling

### Acceptance Criteria

- All blur algorithms tested for energy conservation
- Kernel weights verified
- Mip chain dimensions verified

---

## T-PP-1.3: Color Grading Tests

**File**: `tests/rendering/postprocess/test_color_grading.py`

### Tasks

- [ ] **T-PP-1.3.1**: Test WhiteBalance.apply()
  - Verify temperature to RGB conversion
  - Test daylight (5500K) -> neutral
  - Test tungsten (3200K) -> warm shift
  - Test shade (7500K) -> cool shift

- [ ] **T-PP-1.3.2**: Test LiftGammaGain.apply()
  - Verify lift affects shadows
  - Verify gamma affects midtones
  - Verify gain affects highlights
  - Round-trip: apply then inverse

- [ ] **T-PP-1.3.3**: Test SaturationSettings.apply()
  - saturation=0 -> grayscale
  - saturation=1 -> no change
  - saturation=2 -> double saturation
  - Verify vibrance protects saturated colors

- [ ] **T-PP-1.3.4**: Test LUT3D.load()
  - Load .cube format file
  - Verify grid dimensions parsed
  - Verify domain range parsed

- [ ] **T-PP-1.3.5**: Test LUT3D.sample() trilinear interpolation
  - Sample at grid points -> exact values
  - Sample between points -> interpolated values
  - Verify interpolation weights

- [ ] **T-PP-1.3.6**: Test ChannelMixer.apply()
  - Identity matrix -> no change
  - Swap channels -> correct swapping
  - Zero channel -> channel removed

### Acceptance Criteria

- All color transforms tested
- White balance temperature values verified
- LUT interpolation accuracy verified

---

## T-PP-1.4: Exposure Control Tests

**File**: `tests/rendering/postprocess/test_exposure.py`

### Tasks

- [ ] **T-PP-1.4.1**: Test luminance_to_ev()
  - Verify ISO 12232:2006 formula
  - Known values: 1.0 cd/m2 -> specific EV

- [ ] **T-PP-1.4.2**: Test ev_to_exposure()
  - Verify inverse of luminance_to_ev()
  - Round-trip: L -> EV -> L

- [ ] **T-PP-1.4.3**: Test CenterWeightedMeter.generate_kernel()
  - Verify radial falloff from center
  - Verify kernel sums to 1.0

- [ ] **T-PP-1.4.4**: Test SpotMeter.generate_kernel()
  - Verify small central spot
  - Verify rapid falloff

- [ ] **T-PP-1.4.5**: Test MatrixMeter.generate_kernel()
  - Verify zone weights
  - Verify kernel sums to 1.0

- [ ] **T-PP-1.4.6**: Test HistogramExposure.calculate()
  - Verify percentile calculation
  - Test 50th percentile on uniform histogram

- [ ] **T-PP-1.4.7**: Test EyeAdaptation.update()
  - Verify temporal smoothing
  - Verify asymmetric speeds (bright-to-dark vs dark-to-bright)

### Acceptance Criteria

- ISO 12232:2006 formula verified
- Metering kernels sum to 1.0
- Temporal adaptation verified

---

## T-PP-1.5: DOF Optics Tests

**File**: `tests/rendering/postprocess/test_dof.py`

### Tasks

- [ ] **T-PP-1.5.1**: Test CircleOfConfusion.calculate()
  - Verify formula: CoC from aperture, focal length, focus distance, depth
  - Known optical setup -> expected CoC in pixels

- [ ] **T-PP-1.5.2**: Test CircleOfConfusion.hyperfocal_distance()
  - Verify formula: H = f + f^2 / (N * c)
  - Known f-stop and focal length -> expected hyperfocal

- [ ] **T-PP-1.5.3**: Test BokehShape.generate_kernel() disk
  - Verify circular shape
  - Verify normalized weights

- [ ] **T-PP-1.5.4**: Test BokehShape.generate_kernel() polygon
  - Verify N-sided polygon (5, 6, 7, 8 blades)
  - Verify blade curvature

- [ ] **T-PP-1.5.5**: Test BokehShape.generate_kernel() anamorphic
  - Verify elliptical stretch
  - Verify aspect ratio

- [ ] **T-PP-1.5.6**: Test AutoFocus.update()
  - Verify smooth transition to target
  - Verify transition speed

### Acceptance Criteria

- Optical formulas match published references
- Bokeh kernels are normalized
- All kernel shapes generate correctly

---

## T-PP-1.6: Antialiasing Tests

**File**: `tests/rendering/postprocess/test_antialiasing.py`

### Tasks

- [ ] **T-PP-1.6.1**: Test JitterSequence._halton() base 2
  - index=1 -> 0.5
  - index=2 -> 0.25
  - index=3 -> 0.75
  - Verify first 16 values

- [ ] **T-PP-1.6.2**: Test JitterSequence._halton() base 3
  - index=1 -> 0.333...
  - index=2 -> 0.666...
  - index=3 -> 0.111...
  - Verify first 16 values

- [ ] **T-PP-1.6.3**: Test JitterSequence._generate_halton()
  - Verify (base2, base3) pairs
  - Verify centered to [-0.5, 0.5] range

- [ ] **T-PP-1.6.4**: Test low discrepancy property
  - Generate N samples
  - Verify no clustering
  - Verify coverage of unit square

- [ ] **T-PP-1.6.5**: Test jitter_projection()
  - Verify subpixel offset in projection matrix
  - Verify offset magnitude appropriate for pixel size

### Acceptance Criteria

- Halton sequence values match published tables
- Low discrepancy property verified
- Projection jitter is subpixel

---

## T-PP-1.7: Ambient Occlusion Tests

**File**: `tests/rendering/postprocess/test_ambient_occlusion.py`

### Tasks

- [ ] **T-PP-1.7.1**: Test SSAOKernel.generate() hemisphere
  - All samples have z >= 0
  - Verify sample count

- [ ] **T-PP-1.7.2**: Test SSAOKernel.generate() distribution
  - Distribution is approximately uniform over hemisphere
  - No clustering at poles or equator

- [ ] **T-PP-1.7.3**: Test SSAOKernel.generate() weighting
  - Verify more samples near center (accelerating distribution)
  - Verify weight falloff

- [ ] **T-PP-1.7.4**: Test HBAODirections.generate()
  - Verify N directions uniformly distributed in angle
  - Verify direction count

- [ ] **T-PP-1.7.5**: Test BilateralFilter.weight()
  - Verify spatial Gaussian weight
  - Verify depth-based weight
  - Verify combined formula

### Acceptance Criteria

- All SSAO samples in positive hemisphere
- Distribution is uniform
- Bilateral filter weights correct

---

## T-PP-1.8: Motion Blur Tests

**File**: `tests/rendering/postprocess/test_motion_blur.py`

### Tasks

- [ ] **T-PP-1.8.1**: Test CameraMotionBlur.calculate_velocity()
  - Verify matrix multiplication (prev_VP * curr_VP_inv)
  - Zero motion -> zero velocity
  - Known camera motion -> expected velocity field

- [ ] **T-PP-1.8.2**: Test TileMaxVelocity structure
  - Verify tile dimensions
  - Verify max velocity per tile calculation logic

### Acceptance Criteria

- Camera motion to velocity calculation verified
- Tile structure dimensions correct

---

## Summary

| Task Group | Tasks | Priority |
|------------|-------|----------|
| T-PP-1.1 Tonemapping | 7 | High |
| T-PP-1.2 Bloom | 7 | High |
| T-PP-1.3 Color Grading | 6 | High |
| T-PP-1.4 Exposure | 7 | Medium |
| T-PP-1.5 DOF | 6 | Medium |
| T-PP-1.6 Antialiasing | 5 | Medium |
| T-PP-1.7 Ambient Occlusion | 5 | Low |
| T-PP-1.8 Motion Blur | 2 | Low |

**Total Tasks**: 45
