# PHASE 4 TODO: Material Functions Library

## T-MAT-4.1: Validate Fresnel Functions

**Description**: Verify all Fresnel function variants generate correct GLSL and produce expected results.

**Tasks**:
- [ ] Test `Fresnel(viewDir, normal, power)` code generation
- [ ] Test `FresnelSchlick(cosTheta, F0)` code generation
- [ ] Test `FresnelSchlickRoughness(cosTheta, F0, roughness)` code generation
- [ ] Validate mathematical correctness against reference

**Acceptance Criteria**:
- All Fresnel functions compile without errors
- Output range is [0, 1]
- Edge-on view approaches 1.0
- Face-on view approaches F0

---

## T-MAT-4.2: Validate Normal Blending Functions

**Description**: Ensure normal blending functions correctly combine base and detail normals.

**Tasks**:
- [ ] Test `NormalBlend(base, detail)` whiteout blending
- [ ] Test `NormalBlendRNM(base, detail)` reoriented normal mapping
- [ ] Verify output normals are normalized
- [ ] Compare against reference implementation

**Acceptance Criteria**:
- Blended normals are unit length
- Flat detail normal (0,0,1) returns base unchanged
- Both methods produce visually distinct results
- No artifacts at extreme angles

---

## T-MAT-4.3: Validate Parallax Mapping Functions

**Description**: Verify parallax mapping functions produce correct UV offsets.

**Tasks**:
- [ ] Test `ParallaxOffset` simple offset
- [ ] Test `ParallaxOcclusionMapping` ray marching
- [ ] Verify depth scale parameter works correctly
- [ ] Test edge cases (grazing angles, high depth)

**Acceptance Criteria**:
- UV offset is 0 at normal incidence
- Offset increases with grazing angle
- POM produces more accurate results than simple offset
- No UV discontinuities in common cases

---

## T-MAT-4.4: Validate Triplanar Sampling

**Description**: Ensure triplanar projection produces seamless texturing.

**Tasks**:
- [ ] Test `TriplanarSample` with various sharpness values
- [ ] Verify blending weights sum to 1.0
- [ ] Test on axis-aligned and rotated surfaces
- [ ] Validate no seams at projection boundaries

**Acceptance Criteria**:
- Textures project correctly on all three axes
- Blend weights are smooth
- Sharpness parameter controls transition width
- No visible seams on complex geometry

---

## T-MAT-4.5: Validate Color Operation Functions

**Description**: Verify color space conversions and adjustments are correct.

**Tasks**:
- [ ] Test `sRGBToLinear` gamma decode
- [ ] Test `LinearToSRGB` gamma encode
- [ ] Test `Luminance` calculation
- [ ] Test `Saturation` adjustment
- [ ] Test `Contrast` adjustment

**Acceptance Criteria**:
- sRGB/Linear round-trip preserves color
- Luminance matches ITU-R BT.709 coefficients
- Saturation 0 produces grayscale
- Contrast 0.5 is identity

---

## T-MAT-4.6: Validate Procedural Noise Functions

**Description**: Ensure noise functions produce expected patterns.

**Tasks**:
- [ ] Test `ValueNoise` continuity and range
- [ ] Test `Voronoi` cell generation
- [ ] Test `GradientNoise` smoothness
- [ ] Verify repeatability (same input = same output)

**Acceptance Criteria**:
- Noise output in [0, 1] or [-1, 1] as documented
- No discontinuities in noise field
- Voronoi cells are distinct
- Functions are deterministic

---

## T-MAT-4.7: Validate Procedural Pattern Functions

**Description**: Verify pattern generation functions produce correct output.

**Tasks**:
- [ ] Test `Checkerboard` pattern at various scales
- [ ] Test `RadialGradient` center and radius
- [ ] Verify anti-aliasing behavior
- [ ] Test edge cases (scale = 0, radius = 0)

**Acceptance Criteria**:
- Checkerboard alternates correctly
- RadialGradient falloff is smooth
- Scale parameter works as expected
- No artifacts at extreme parameters

---

## T-MAT-4.8: Validate Masking Functions

**Description**: Ensure masking functions produce correct falloff patterns.

**Tasks**:
- [ ] Test `BoxMask` with various falloff values
- [ ] Test `SphereMask` with various falloff values
- [ ] Verify output range [0, 1]
- [ ] Test boundary behavior

**Acceptance Criteria**:
- Inside region is 1.0
- Outside region is 0.0
- Falloff produces smooth transition
- Works in 2D and 3D

---

## T-MAT-4.9: Validate Blend Mode Functions

**Description**: Verify blend mode functions match standard definitions.

**Tasks**:
- [ ] Test `BlendOverlay` against Photoshop reference
- [ ] Test `BlendSoftLight` against Photoshop reference
- [ ] Verify symmetry properties
- [ ] Test with various input ranges

**Acceptance Criteria**:
- BlendOverlay matches standard overlay formula
- BlendSoftLight matches standard soft light formula
- Results are in [0, 1] for inputs in [0, 1]
- No color banding or artifacts

---

## T-MAT-4.10: Function Dependency Resolution

**Description**: Verify the dependency system correctly orders function inclusion.

**Tasks**:
- [ ] Test function with no dependencies
- [ ] Test function with single dependency
- [ ] Test function chain (A depends on B depends on C)
- [ ] Test shared dependencies (A and B both depend on C)

**Acceptance Criteria**:
- Functions included in dependency order
- No duplicate function definitions
- Circular dependencies detected and rejected
- Only needed functions are included
