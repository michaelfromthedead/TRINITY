# PHASE 5 TODO: Advanced Shading Models

## T-MAT-5.1: Validate SubsurfaceScattering Model

**Description**: Verify SSS implementation with Burley diffusion profiles.

**Tasks**:
- [ ] Test SSS parameter creation and validation
- [ ] Test preset application (Skin, Wax, Jade, Milk)
- [ ] Verify scattering_radius units are correct
- [ ] Test integration with base PBR material

**Acceptance Criteria**:
- SSS parameters validate correctly
- Presets produce distinct visual results
- Scattering radius affects diffusion extent
- Model can be combined with base PBR

---

## T-MAT-5.2: Validate ClearCoat Model

**Description**: Ensure clearcoat secondary specular layer works correctly.

**Tasks**:
- [ ] Test clearcoat parameter creation
- [ ] Verify IOR-based F0 calculation
- [ ] Test clearcoat_normal separate from base normal
- [ ] Validate intensity range [0, 1]

**Acceptance Criteria**:
- Clearcoat adds visible secondary highlight
- F0 calculated from IOR (1.5 -> ~0.04)
- Clearcoat normal can differ from base normal
- Intensity 0 disables the effect

---

## T-MAT-5.3: Validate Anisotropy Model

**Description**: Verify anisotropic roughness with GGX parameterization.

**Tasks**:
- [ ] Test anisotropy parameter creation
- [ ] Verify rotation parameter affects highlight direction
- [ ] Test with tangent map input
- [ ] Validate strength range [-1, 1]

**Acceptance Criteria**:
- Highlights stretch in anisotropy direction
- Rotation parameter rotates highlight correctly
- Tangent map overrides default tangent
- Negative values rotate 90 degrees

---

## T-MAT-5.4: Validate Sheen Model

**Description**: Ensure sheen effect produces fabric/velvet appearance.

**Tasks**:
- [ ] Test sheen parameter creation
- [ ] Verify sheen_color affects edge coloring
- [ ] Test sheen_roughness affects softness
- [ ] Validate integration with base color

**Acceptance Criteria**:
- Sheen adds soft edge highlights
- Sheen color is visible at grazing angles
- Higher roughness produces softer effect
- Effect is energy-conserving

---

## T-MAT-5.5: Validate Iridescence Model

**Description**: Verify thin film interference produces correct color shifting.

**Tasks**:
- [ ] Test iridescence parameter creation
- [ ] Verify thickness affects color wavelength
- [ ] Test iridescence_ior affects pattern
- [ ] Validate thickness map support

**Acceptance Criteria**:
- Color shifts with viewing angle
- Different thicknesses produce different colors
- IOR affects the color pattern spacing
- Thickness map creates varied iridescence

---

## T-MAT-5.6: Validate Transmission Model

**Description**: Ensure transmission/refraction works with Beer-Lambert attenuation.

**Tasks**:
- [ ] Test transmission parameter creation
- [ ] Verify attenuation_color affects thick regions
- [ ] Test attenuation_distance units
- [ ] Validate thickness parameter

**Acceptance Criteria**:
- Transmission allows light through material
- Thick regions are more colored (Beer-Lambert)
- Attenuation distance controls color density
- Thickness affects attenuation amount

---

## T-MAT-5.7: Model Combination Testing

**Description**: Verify advanced models can be combined correctly.

**Tasks**:
- [ ] Test clearcoat + anisotropy (car paint)
- [ ] Test SSS + transmission (wax)
- [ ] Test iridescence + clearcoat (soap bubbles)
- [ ] Verify energy conservation when combined

**Acceptance Criteria**:
- Combined models produce expected appearance
- No visual artifacts from combination
- Energy is conserved (not over-bright)
- Each model's contribution is visible

---

## T-MAT-5.8: Preset Library Validation

**Description**: Verify all presets produce correct results.

**Tasks**:
- [ ] Test all SSS presets (Skin, Wax, Jade, Milk)
- [ ] Document expected visual appearance
- [ ] Verify preset parameter values
- [ ] Test preset modification workflow

**Acceptance Criteria**:
- Each preset produces documented appearance
- Presets are visually distinct
- Parameters are physically plausible
- Presets can be used as starting points

---

## T-MAT-5.9: Performance Profiling

**Description**: Measure GPU cost of each advanced model.

**Tasks**:
- [ ] Profile SSS blur pass cost
- [ ] Profile transmission refraction cost
- [ ] Profile clearcoat, anisotropy, sheen, iridescence
- [ ] Document performance tier recommendations

**Acceptance Criteria**:
- Performance numbers documented per model
- Platform-specific recommendations provided
- Graceful degradation path defined
- No unexpected performance cliffs

---

## T-MAT-5.10: Fallback System

**Description**: Implement graceful degradation for expensive models.

**Tasks**:
- [ ] Define fallback for SSS (diffuse approximation)
- [ ] Define fallback for transmission (alpha blend)
- [ ] Define fallback for iridescence (disabled)
- [ ] Test automatic tier selection

**Acceptance Criteria**:
- Fallbacks produce acceptable approximations
- No visual discontinuities at quality transitions
- Performance targets are met with fallbacks
- Fallback selection is automatic per platform
