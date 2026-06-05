# RECOMMENDATIONS: engine/rendering/lighting

## Rust Bridge Requirements

### High Priority (Blocking Core Rendering)

| Requirement | Description | Complexity |
|-------------|-------------|------------|
| Shadow Texture Handles | Expose wgpu texture creation for shadow maps (2D, cube, array) | Medium |
| Shadow Depth Pass | Create render pass for writing shadow depth from Python scene data | High |
| Froxel Buffer Upload | GPU buffer for froxel light indices and counts | Medium |
| Light Data Upload | Structured buffer for light parameters (position, color, attenuation) | Low |
| Shadow Sampler | Expose comparison sampler for depth testing | Low |

### Medium Priority (Quality Features)

| Requirement | Description | Complexity |
|-------------|-------------|------------|
| Cubemap Handles | Expose cubemap texture for reflection probes and sky | Medium |
| SH Coefficient Upload | GPU buffer for SH L2 coefficients (27 floats per probe) | Low |
| Probe Atlas | Texture atlas for irradiance/visibility probe data | Medium |
| Lightmap Texture | 2D texture array for baked lightmaps | Medium |
| IES Texture | 1D or 2D texture for IES profile lookup | Low |

### Low Priority (Advanced Features)

| Requirement | Description | Complexity |
|-------------|-------------|------------|
| Ray Tracing Interface | Expose hardware RT or BVH traversal for DDGI | High |
| Screen-Space Buffer | Access to depth/normal for contact shadows | Medium |
| LTC LUT Texture | Precomputed LTC matrices for area lights | Low |
| Temporal History | Previous frame data for temporal filtering | Medium |

## Integration Strategy

### Phase 1: Basic Shadows (Weeks 1-2)
1. Create shadow atlas texture in Rust (2D depth texture array)
2. Implement CSM depth render pass via frame graph
3. Expose `ShadowAtlas` handle to Python
4. Create basic PCF sampling shader in WGSL
5. Wire directional light shadows through bridge

**Validation:** Render scene with hard shadows from directional light

### Phase 2: Clustered Lighting (Weeks 3-4)
1. Create froxel grid compute shader
2. Upload light data to structured buffer
3. Implement light assignment compute pass
4. Create clustered shading in fragment shader
5. Support point and spot lights

**Validation:** Render scene with 100+ dynamic lights

### Phase 3: Soft Shadows (Weeks 5-6)
1. Implement PCSS shader with blocker search
2. Add VSM shadow map generation (moments)
3. Create ESM shader variant
4. Expose filter selection to Python

**Validation:** Compare shadow quality across filter types

### Phase 4: Global Illumination (Weeks 7-10)
1. Implement SH evaluation shader
2. Create probe grid interpolation
3. Add DDGI probe update pass (software RT fallback)
4. Implement octahedral atlas sampling
5. Add reflection probe cubemap sampling

**Validation:** Render scene with indirect lighting and reflections

## Testing Strategy

### Unit Tests (Python)
- Shadow split computation accuracy
- SH coefficient generation
- Octahedral encoding round-trip
- Froxel index calculation
- Light culling correctness

### Integration Tests (Rust + Python)
- Shadow texture creation and binding
- Buffer upload verification
- Render pass execution
- Frame graph integration

### Visual Tests (Golden Image)
- Shadow map depth output
- Cascade visualization
- Froxel grid debug view
- Probe grid interpolation
- GI comparison (baked vs dynamic)

### Performance Tests
- Froxel culling with 1000 lights
- Shadow map rendering at 4K
- DDGI update frequency impact
- Memory usage for probe atlases

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ray tracing unavailable | DDGI non-functional | Software BVH fallback, prioritize baked probes |
| Shadow memory pressure | Atlas overflow at high quality | Implement dynamic resolution, prioritize visible cascades |
| Froxel overhead | CPU/GPU sync stalls | Async compute, double-buffer froxel data |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| VSM light bleeding | Visual artifacts | Provide ESM alternative, tunable parameters |
| SH ringing | Overbrightening at L2 | Windowing function, L1 fallback |
| Probe pop-in | Visible interpolation | Temporal blending, higher probe density |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| IES profile unsupported | Reduced photometric accuracy | Default to point light attenuation |
| Area light complexity | LTC integration work | Defer to point light approximation |
| Contact shadow cost | Performance on low-end | Make optional, screen-space LOD |

## Recommended Rust Bridge API

```rust
// Shadow map management
pub fn create_shadow_atlas(width: u32, height: u32, layers: u32) -> TextureHandle;
pub fn allocate_shadow_region(atlas: TextureHandle, w: u32, h: u32) -> ShadowRegion;
pub fn render_shadow_pass(atlas: TextureHandle, region: ShadowRegion, view_proj: Mat4, scene: SceneRef);

// Light data
pub fn create_light_buffer(max_lights: u32) -> BufferHandle;
pub fn upload_lights(buffer: BufferHandle, lights: &[LightData]);

// Froxel grid
pub fn create_froxel_grid(tiles_x: u32, tiles_y: u32, slices: u32) -> FroxelGridHandle;
pub fn update_froxel_grid(grid: FroxelGridHandle, lights: BufferHandle, camera: CameraData);

// GI probes
pub fn create_probe_atlas(probe_size: u32, probe_count: u32) -> TextureHandle;
pub fn update_ddgi_probes(atlas: TextureHandle, rays_per_probe: u32, scene: SceneRef);

// Cubemaps
pub fn create_cubemap(size: u32, mip_levels: u32) -> TextureHandle;
pub fn sample_cubemap(cubemap: TextureHandle, direction: Vec3, roughness: f32) -> Vec3;
```

## Priority Matrix

| Feature | User Impact | Implementation Effort | Priority Score |
|---------|-------------|----------------------|----------------|
| CSM Shadows | Critical | Medium | 1 |
| PCF Filtering | High | Low | 2 |
| Clustered Lighting | High | Medium | 3 |
| Point/Spot Shadows | High | Medium | 4 |
| PCSS Soft Shadows | Medium | Medium | 5 |
| SH Light Probes | Medium | Low | 6 |
| Reflection Probes | Medium | Medium | 7 |
| DDGI | Medium | High | 8 |
| VSM/ESM | Low | Low | 9 |
| Area Lights | Low | High | 10 |
| Contact Shadows | Low | Medium | 11 |
