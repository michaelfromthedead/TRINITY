# PHASE 3 ARCH: Shader Implementation

## Objective

Implement GPU shaders (WGSL/HLSL) for all post-processing effects. The Python CPU implementations serve as the authoritative specification for shader behavior.

## Architecture Decisions

### AD-1: Primary Shader Language

**Decision**: WGSL as primary, HLSL as secondary target.

**Rationale**: WGSL is the WebGPU standard and aligns with the engine's cross-platform goals. HLSL can be transpiled from WGSL if needed for DirectX targets.

### AD-2: Compute vs Graphics Shaders

**Decision**: Use compute shaders for all post-processing effects.

**Rationale**: 
- Compute shaders provide more flexibility (arbitrary dispatch sizes)
- Better cache utilization with explicit group sizes
- Consistent pattern across all effects
- Graphics shaders only where rasterization benefits apply (none here)

### AD-3: Shader Organization

**Decision**: One shader file per effect category with multiple entry points.

**Rationale**: Related algorithms share common functions (e.g., all tonemap operators share color space transforms).

**Structure**:
```
shaders/
  postprocess/
    tonemapping.wgsl      # 8 tonemap operators
    bloom.wgsl            # threshold, blur, upsample
    color_grading.wgsl    # all color transforms
    antialiasing.wgsl     # FXAA, TAA
    ambient_occlusion.wgsl # SSAO, HBAO, GTAO
    dof.wgsl              # CoC, blur, composite
    exposure.wgsl         # histogram, adaptation
    motion_blur.wgsl      # velocity, blur
    upscaling.wgsl        # bilinear, FSR1
```

### AD-4: Common Functions Library

**Decision**: Factor common functions into shared include file.

**Rationale**: Color space conversions, sampling utilities, and math functions are shared across effects.

**Content**:
```wgsl
// common.wgsl
fn srgb_to_linear(c: vec3f) -> vec3f
fn linear_to_srgb(c: vec3f) -> vec3f
fn rgb_to_luminance(c: vec3f) -> f32
fn sample_bilinear(tex: texture_2d<f32>, uv: vec2f) -> vec4f
```

### AD-5: Bind Group Layout Consistency

**Decision**: Standardize bind group layout across effects.

**Rationale**: Consistent layout simplifies pipeline management and resource binding.

**Standard Layout**:
```
Group 0: Per-frame uniforms (camera, time, resolution)
Group 1: Effect-specific uniforms (settings)
Group 2: Input textures
Group 3: Output textures (UAVs)
```

### AD-6: Thread Group Sizing

**Decision**: 8x8 thread groups for 2D image processing, 64x1x1 for 1D operations.

**Rationale**: 8x8=64 threads matches common wave size. Provides good cache locality for 2D sampling patterns.

### AD-7: Algorithm Fidelity

**Decision**: Shaders must match Python reference implementations exactly.

**Rationale**: The Python code is the specification. Shaders are the optimized implementation. Results must be visually identical.

**Verification**: Side-by-side comparison with CPU fallback path.

## Shader Specifications

### Tonemapping Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| tonemap_reinhard | L / (1 + L) | Reinhard.tonemap_value() |
| tonemap_reinhard_extended | With white point | ReinhardExtended.tonemap_value() |
| tonemap_aces | sRGB->ACEScg->RRT+ODT | ACES.tonemap_value() |
| tonemap_aces_fitted | Krzysztof Narkowicz fit | ACESFitted._rrt_odt() |
| tonemap_agx | Log encoding + look | AgX.tonemap_value() |
| tonemap_filmic | Hable curve | Filmic.tonemap_value() |
| tonemap_custom | Artist curve | CustomCurve._hermite_interpolate() |

### Bloom Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| bloom_threshold | Soft-knee threshold | BloomThreshold.apply() |
| bloom_downsample | 13-tap filter | BloomDownsample |
| bloom_blur_h | Horizontal Gaussian | _gaussian_blur() horizontal |
| bloom_blur_v | Vertical Gaussian | _gaussian_blur() vertical |
| bloom_upsample | Bilinear + blend | BloomUpsample |

### Color Grading Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| apply_white_balance | Temperature/tint matrix | WhiteBalance.apply() |
| apply_lgg | Lift/gamma/gain | LiftGammaGain.apply() |
| apply_saturation | With vibrance | SaturationSettings.apply() |
| sample_lut3d | Trilinear LUT | LUT3D.sample() |
| apply_channel_mixer | Matrix multiply | ChannelMixer.apply() |

### TAA Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| taa_resolve | Reproject + clamp + blend | TAA.apply() |
| taa_neighborhood_clamp | AABB clamping | - |
| taa_velocity_sample | Motion vector lookup | - |

### SSAO Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| ssao_main | Hemisphere sampling | SSAO.calculate() |
| hbao_main | Horizon-based AO | HBAO.calculate() |
| gtao_main | Ground truth AO | GTAO.calculate() |
| bilateral_blur | Depth-aware blur | BilateralFilter |

### DOF Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| calculate_coc | From depth | CircleOfConfusion.calculate() |
| dof_blur_near | Near field | NearFieldDOF.blur() |
| dof_blur_far | Far field | FarFieldDOF.blur() |
| dof_composite | Combine layers | DOFComposite |

### Exposure Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| histogram_clear | Zero histogram | - |
| histogram_bin | Per-pixel binning | - |
| histogram_reduce | Sum to average | - |
| eye_adaptation | Temporal smoothing | EyeAdaptation.update() |

### Motion Blur Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| tile_max | Max velocity per tile | TileMaxVelocity |
| tile_neighbor | Neighbor max | - |
| motion_blur_gather | Sample along velocity | apply_blur() |

### Upscaling Shaders

| Entry Point | Algorithm | Python Reference |
|-------------|-----------|------------------|
| upscale_bilinear | Simple bilinear | - |
| fsr1_easu | Edge-aware upscale | - |
| fsr1_rcas | Sharpening | - |

## Dependencies

### Required Engine Components

| Component | Purpose |
|-----------|---------|
| Shader compiler | WGSL -> SPIR-V |
| Bind group layout factory | Consistent layouts |
| Pipeline cache | Avoid recompilation |

### External References

| Algorithm | Reference |
|-----------|-----------|
| ACES | Academy Color Encoding System specification |
| FSR1 | AMD FidelityFX Super Resolution 1.0 |
| GTAO | Ground Truth Ambient Occlusion (Activision) |
| AgX | Troy Sobotka's AgX look |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Shader precision mismatch | Medium | Test against CPU reference |
| Platform-specific bugs | Medium | Test on multiple GPUs |
| Performance not meeting targets | Medium | Profile and optimize hot paths |
| WGSL features missing | Low | Fall back to more verbose code |

## Deliverables

1. WGSL shader files for all post-processing effects
2. Shared common.wgsl with utility functions
3. HLSL equivalents (if needed for DirectX)
4. Test harness comparing GPU output to CPU reference
5. Performance benchmarks per effect
