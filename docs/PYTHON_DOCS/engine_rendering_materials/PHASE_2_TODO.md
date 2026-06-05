# PHASE 2 TODO: PBR Model and Texture Pipeline

## T-MAT-2.1: Validate PBRParameters Dataclass

**Description**: Verify PBRParameters correctly validates and stores PBR material properties.

**Tasks**:
- [ ] Test creation with valid parameter ranges
- [ ] Test clamping behavior for out-of-range values
- [ ] Verify all PBR fields are accessible (base_color, metallic, roughness, normal, emissive, ao, opacity)
- [ ] Test serialization/deserialization

**Acceptance Criteria**:
- Metallic and roughness clamped to [0, 1]
- Base color accepts RGB or RGBA
- Invalid values raise or clamp without corruption
- Parameters survive round-trip serialization

---

## T-MAT-2.2: Validate PBRMaterial Component

**Description**: Confirm PBRMaterial integrates dirty flags and change callbacks.

**Tasks**:
- [ ] Test PBRMaterial creation and parameter access
- [ ] Verify dirty flag setting on parameter change
- [ ] Test change callback invocation
- [ ] Validate tracked descriptor updates

**Acceptance Criteria**:
- Parameter changes set PARAMETERS dirty flag
- Registered callbacks are invoked on change
- Descriptors reflect current parameter state
- Multiple callbacks can be registered

---

## T-MAT-2.3: Validate PBRTextureSet Binding

**Description**: Ensure PBRTextureSet correctly maps textures to shader slots.

**Tasks**:
- [ ] Test texture assignment to slots (base_color, normal, orm, emissive)
- [ ] Verify channel configuration for packed textures
- [ ] Test dirty flag on texture rebind
- [ ] Validate slot indices match shader expectations

**Acceptance Criteria**:
- Textures bind to correct slots (0-3)
- Packed ORM texture channels are correctly swizzled
- TEXTURES dirty flag set on rebind
- Missing textures use fallback (white, flat normal, etc.)

---

## T-MAT-2.4: Validate PBRWorkflow Selection

**Description**: Verify workflow enum affects shader generation appropriately.

**Tasks**:
- [ ] Test METALLIC_ROUGHNESS workflow
- [ ] Test SPECULAR_GLOSSINESS workflow
- [ ] Verify shader uniforms differ by workflow
- [ ] Test workflow switching on existing material

**Acceptance Criteria**:
- Metallic-roughness uses metallic/roughness uniforms
- Specular-glossiness uses specular/glossiness uniforms
- Workflow change sets SHADER dirty flag
- Generated shader reflects workflow choice

---

## T-MAT-2.5: Integration with Texture Table

**Description**: Connect PBRTextureSet to the GPU texture table system.

**Tasks**:
- [ ] Test texture registration in texture table
- [ ] Verify bindless handle retrieval
- [ ] Test texture update propagation
- [ ] Validate GPU-side binding

**Acceptance Criteria**:
- PBR textures register in texture table on material creation
- Bindless handles are valid and usable in shaders
- Texture updates propagate to GPU
- No texture slot conflicts with other systems

---

## T-MAT-2.6: Fallback Texture Generation

**Description**: Ensure fallback textures are generated for missing PBR inputs.

**Tasks**:
- [ ] Implement/test white fallback for base color
- [ ] Implement/test flat normal fallback (0.5, 0.5, 1.0)
- [ ] Implement/test default ORM (1.0, 0.5, 0.0)
- [ ] Implement/test black fallback for emissive

**Acceptance Criteria**:
- Materials render without explicit textures
- Fallbacks produce physically plausible results
- Fallbacks are cached and reused
- No visual artifacts from missing textures
