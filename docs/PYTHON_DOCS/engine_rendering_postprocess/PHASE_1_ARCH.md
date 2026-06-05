# PHASE 1 ARCH: CPU Algorithm Verification

## Objective

Verify all existing CPU-side mathematical implementations through unit testing before proceeding to GPU integration. This phase creates a regression test suite that serves as the specification for future shader implementations.

## Architecture Decisions

### AD-1: Test Structure Mirroring Source

**Decision**: Create test files that mirror source file structure.

**Rationale**: Each source file contains related algorithms. Test files should maintain this organization for maintainability.

**Structure**:
```
tests/
  rendering/
    postprocess/
      test_tonemapping.py
      test_bloom.py
      test_color_grading.py
      test_exposure.py
      test_dof.py
      test_antialiasing.py
      test_ambient_occlusion.py
      test_motion_blur.py
```

### AD-2: Known-Answer Tests for Tonemap Operators

**Decision**: Use published reference values or hand-calculated values for tonemap operator tests.

**Rationale**: Tonemap operators have well-defined mathematical formulas. Tests should verify the implementation matches the formula, not just that it runs.

**Approach**:
- ACES: Use official ACES test patterns
- Reinhard: Calculate expected output from formula: `L / (1 + L)`
- AgX: Verify log encoding and look transform against reference
- Filmic: Verify shoulder/toe curves match John Hable's published formula

### AD-3: Energy Conservation Tests for Blur Algorithms

**Decision**: Verify blur algorithms conserve energy (sum of output equals sum of input for normalized kernels).

**Rationale**: Blur algorithms should not create or destroy light. Energy conservation is a fundamental correctness property.

**Tests**:
- Gaussian blur: verify kernel sums to 1.0, output energy matches input
- Kawase blur: verify 5-point weights correct
- Box blur: verify uniform weights

### AD-4: Round-Trip Tests for Color Transforms

**Decision**: Verify invertible color transforms round-trip correctly.

**Rationale**: White balance, lift/gamma/gain, and channel mixing are reversible. Round-trip tests catch numerical precision issues.

**Tests**:
- Apply transform then inverse, verify within epsilon
- Edge cases: extreme temperature values, zero saturation

### AD-5: Optical Formula Verification for DOF

**Decision**: Verify DOF calculations against published optical formulas.

**Rationale**: Circle of Confusion and hyperfocal distance have exact formulas from optics literature. Implementation must match.

**Reference**: Sidney Ray, "Applied Photographic Optics", 3rd edition formulas for CoC and hyperfocal distance.

### AD-6: Low-Discrepancy Verification for Halton Sequence

**Decision**: Verify Halton sequence has expected low-discrepancy properties.

**Rationale**: Halton sequences are used for TAA jitter. Incorrect implementation leads to visible banding artifacts.

**Tests**:
- First N values match known Halton values
- Star discrepancy metric within expected bounds
- No duplicate values in sequence

### AD-7: Hemisphere Distribution for SSAO Kernels

**Decision**: Verify SSAO hemisphere kernels have uniform distribution.

**Rationale**: Biased kernel distribution causes visible AO artifacts.

**Tests**:
- All samples in positive hemisphere (z >= 0)
- Distribution is approximately uniform over hemisphere
- Sample weighting correct (more samples near center)

## Components Under Test

### Tonemapping (tonemapping.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| Reinhard.tonemap_value() | Known-answer | `1.0 -> 0.5`, `4.0 -> 0.8` |
| ReinhardExtended.tonemap_value() | Known-answer | Verify white point handling |
| ACES.tonemap_value() | Known-answer | Match official ACES reference |
| ACESFitted._rrt_odt() | Known-answer | Verify approximation accuracy |
| AgX.tonemap_value() | Known-answer | Log encoding + look transform |
| Filmic.tonemap_value() | Known-answer | Shoulder/toe S-curve |
| CustomCurve._hermite_interpolate() | Property | Smooth interpolation, endpoint matching |

### Bloom (bloom.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| BloomThreshold.apply() | Known-answer | Soft-knee threshold curve |
| BloomDownsample.generate_mip_chain() | Property | Correct mip dimensions |
| _gaussian_blur() | Energy conservation | Output energy = input energy |
| _kawase_blur() | Pattern | 5-point cross pattern |
| _box_blur() | Energy conservation | Uniform averaging |

### Color Grading (color_grading.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| WhiteBalance.apply() | Known-answer | Temperature/tint to RGB matrix |
| LiftGammaGain.apply() | Round-trip | Invertible transform |
| SaturationSettings.apply() | Boundary | 0 saturation -> grayscale |
| LUT3D.sample() | Interpolation | Trilinear interpolation correct |
| ChannelMixer.apply() | Identity | Identity matrix = no change |

### Exposure (exposure.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| luminance_to_ev() | Known-answer | ISO 12232:2006 formula |
| ev_to_exposure() | Round-trip | Inverse of luminance_to_ev() |
| CenterWeightedMeter.generate_kernel() | Property | Radial falloff |
| HistogramExposure.calculate() | Property | Percentile calculation |
| EyeAdaptation.update() | Temporal | Smooth transition |

### DOF (dof.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| CircleOfConfusion.calculate() | Known-answer | Optical formula |
| CircleOfConfusion.hyperfocal_distance() | Known-answer | Optical formula |
| BokehShape.generate_kernel() | Property | Normalized weights |
| BokehShape.generate_polygon() | Property | N-sided polygon |

### Antialiasing (antialiasing.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| JitterSequence._halton() | Known-answer | Base-2 and base-3 sequences |
| JitterSequence._generate_halton() | Property | Low discrepancy |
| jitter_projection() | Property | Small subpixel offset |

### Ambient Occlusion (ambient_occlusion.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| SSAOKernel.generate() | Property | Hemisphere distribution |
| SSAOKernel.generate() | Boundary | All z >= 0 |
| HBAODirections.generate() | Property | Uniform angle distribution |
| BilateralFilter.weight() | Known-answer | Gaussian * depth weight |

### Motion Blur (motion_blur.py)

| Component | Test Type | Expected Behavior |
|-----------|-----------|-------------------|
| CameraMotionBlur.calculate_velocity() | Known-answer | Matrix multiply |
| TileMaxVelocity structure | Property | Correct tile dimensions |

## Dependencies

- pytest for test framework
- numpy for numerical comparisons (optional, can use pure Python)
- No GPU required
- No external assets required

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Tests pass but formulas wrong | Use published reference values, not just "it runs" |
| Floating point precision | Use relative epsilon comparisons |
| Edge cases missed | Explicit boundary value tests |

## Deliverables

1. Test files for all 8 modules
2. Test data files with known-answer inputs/outputs
3. Test coverage report showing algorithm coverage
4. Documentation of any discovered bugs in existing implementations
