# RECOMMENDATIONS: engine/rendering/postprocess

## Rust Bridge Requirements

### High Priority

| Requirement | Description | Complexity | Rationale |
|-------------|-------------|------------|-----------|
| Effect Settings Bridge | PyO3 bindings for all effect settings structs | Medium | Python controls parameters, Rust executes |
| PostProcessStack State | Sync stack ordering, active effects, quality preset | Medium | Rust needs to know which effects to run |
| Tonemap Operator Bridge | Pass operator type and parameters | Low | 8 operators, simple enum + parameters |
| Color Grading LUT Upload | Transfer 3D LUT data to GPU texture | Medium | .cube parsing done in Python, upload in Rust |
| Bloom Settings Bridge | Threshold, scatter, intensity per mip level | Low | Well-defined numeric parameters |

### Medium Priority

| Requirement | Description | Complexity | Rationale |
|-------------|-------------|------------|-----------|
| Exposure State Bridge | Eye adaptation history, current EV | Medium | Temporal state must persist across frames |
| DOF Parameters Bridge | CoC settings, bokeh shape, focus distance | Medium | Autofocus state is temporal |
| TAA History Bridge | History buffer handles, jitter index | High | Complex temporal state management |
| PostProcessVolume Blend | Blended settings from multiple volumes | Medium | Camera position -> interpolated settings |
| Quality Preset Application | Apply preset overrides at bridge layer | Low | Simple configuration mapping |

### Low Priority

| Requirement | Description | Complexity | Rationale |
|-------------|-------------|------------|-----------|
| AO Settings Bridge | Kernel type, radius, bias | Low | Straightforward parameters |
| Motion Blur Settings | Velocity scale, tile size | Low | Simple numeric parameters |
| Upscaler Selection | FSR/DLSS/XeSS mode and quality | Low | Enum selection |
| Execution Flags | SKIP_IF_DISABLED, FORCE_ASYNC | Low | Per-frame boolean flags |

## Integration Strategy

### Phase 1: Core Image Processing (Weeks 1-3)

1. **Tonemapping Compute Shader**
   - Port all 8 operators to single WGSL shader with switch
   - Bind tonemapping settings uniform buffer
   - Simple fullscreen dispatch

2. **Color Grading Compute Shader**
   - White balance, LGG, saturation in single pass
   - 3D LUT texture sampling with trilinear
   - Chain with tonemapping or separate pass

3. **Settings Bridge Foundation**
   - Create Rust structs mirroring Python dataclasses
   - PyO3 `#[pyclass]` for cross-language access
   - Simple synchronization pattern

### Phase 2: Bloom Pipeline (Weeks 4-5)

1. **Threshold Pass**
   - Soft knee extraction to mip 0
   - HDR input, HDR output

2. **Downsample Chain**
   - Generate mips 1-N
   - Box filter or bilinear

3. **Blur Passes**
   - Separable Gaussian per mip
   - Kawase as alternative

4. **Upsample + Composite**
   - Additive blend up the chain
   - Final composite with scene

### Phase 3: Exposure & Adaptation (Weeks 6-7)

1. **Histogram Compute**
   - Log-luminance histogram
   - Parallel reduction

2. **Auto Exposure**
   - Percentile-based from histogram
   - Metering weight mask

3. **Eye Adaptation**
   - Temporal smoothing compute
   - Read/write history buffer

### Phase 4: Temporal Effects (Weeks 8-10)

1. **TAA Pipeline**
   - Jitter application (already REAL)
   - Motion vector generation (if not from gbuffer)
   - History reprojection
   - Neighborhood clamp
   - History blend

2. **Motion Blur**
   - Tile velocity max
   - Directional blur based on velocity

### Phase 5: Depth Effects (Weeks 11-12)

1. **DOF**
   - CoC buffer generation
   - Gather bokeh or scatter
   - Near/far field separation

2. **Ambient Occlusion**
   - GTAO or HBAO compute
   - Bilateral blur
   - Composite to scene

## Testing Strategy

### Unit Tests (Python Side)

| Test Area | Approach | Coverage |
|-----------|----------|----------|
| Tonemap Operators | Reference values for known inputs | All 8 operators |
| CoC Calculation | Known aperture/focal length -> expected CoC | Edge cases (infinity, macro) |
| Halton Sequence | Verify low-discrepancy properties | First 64 samples |
| Bloom Gaussian | Compare to NumPy convolution | Various kernel sizes |
| LUT Interpolation | Verify trilinear correctness | Corner and center values |
| Eye Adaptation | Temporal smoothing curve | Step response |

### Integration Tests (Rust Side)

| Test Area | Approach | Coverage |
|-----------|----------|----------|
| Settings Roundtrip | Python -> Rust -> verify | All settings structs |
| LUT Upload | Upload -> sample -> verify | 16x16x16 and 64x64x64 |
| Shader Compilation | Validate all WGSL | Compilation success |
| Compute Dispatch | Dispatch -> readback -> verify | Basic execution |

### Visual Tests (Golden Images)

| Test Area | Approach | Coverage |
|-----------|----------|----------|
| Tonemapping | HDR test image -> LDR reference | All operators |
| Bloom | High-contrast scene | Threshold, blur, composite |
| Color Grading | Neutral image + extreme settings | Each adjustment type |
| DOF | Depth gradient + focus plane | CoC visualization |

### Performance Tests

| Test Area | Approach | Target |
|-----------|----------|--------|
| Full Stack 1080p | Measure frame time | < 2ms |
| Full Stack 4K | Measure frame time | < 4ms |
| Bloom (8 mips) | Individual timing | < 0.5ms |
| TAA | Individual timing | < 0.3ms |

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TAA ghosting/artifacts | High | Medium | Start with simple blend, add clamping incrementally |
| Bloom energy conservation | Medium | Low | Validate sum of blur weights equals 1.0 |
| LUT precision loss | Low | Medium | Use R16G16B16A16 format for LUT texture |
| Eye adaptation instability | Medium | Medium | Clamp adaptation rate, use exponential smoothing |
| DOF performance on scatter | High | High | Use gather-based approach, fallback to simpler |

### Architectural Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Settings sync drift | Medium | High | Single source of truth pattern, version stamps |
| Effect ordering bugs | Low | Medium | Validate order in tests, assert dependencies |
| Memory pressure from history buffers | Medium | Medium | Pool buffers, lazy allocation |
| Async compute synchronization | High | Medium | Start with graphics queue only, add async later |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Shader debugging time | High | Medium | Invest in RenderDoc integration early |
| TAA complexity underestimated | Medium | High | Allow buffer week, prioritize stability over quality |
| Vendor SDK integration (FSR2/DLSS) | Medium | Low | Defer to Phase 6, use simpler upscaling first |

## Dependencies

### External Dependencies

| Dependency | Status | Required For |
|------------|--------|--------------|
| wgpu | Available | All compute shaders |
| PyO3 | Available | Settings bridge |
| naga | Available | WGSL validation |

### Internal Dependencies

| Dependency | Status | Required For |
|------------|--------|--------------|
| Frame Graph | Available | Pass scheduling |
| Resource Manager | Available | Target allocation |
| RHI Command Lists | Needed | Dispatch recording |
| Motion Vectors | Needed | TAA, motion blur |
| Depth Buffer | Needed | DOF, AO |
| GBuffer Normals | Needed | AO |
