# PHASE 4 TODO: Global Illumination

## T-LGT-4.1: SH Probe GPU Buffer

**Description:** Create storage buffer for spherical harmonics probe data.

**Tasks:**
- [ ] Define `SHProbeGPU` struct: 7 vec4s (27 coefficients + 1 padding)
- [ ] Add `SphericalHarmonics.to_gpu_format()` returning packed floats
- [ ] Add `ProbeGrid.to_gpu_buffer()` returning buffer with all probes
- [ ] Include grid metadata: origin, spacing, dimensions
- [ ] Upload buffer via renderer-backend

**Acceptance Criteria:**
- [ ] Buffer contains all probe coefficients in correct order
- [ ] Grid metadata available in uniform buffer
- [ ] WGSL can read probe at arbitrary index
- [ ] Unit test validates coefficient round-trip

---

## T-LGT-4.2: SH Evaluation Shader

**Description:** Implement L2 spherical harmonics evaluation in WGSL.

**Tasks:**
- [ ] Create `shaders/gi_sh.wgsl` with SH functions
- [ ] Implement `evaluate_sh_l2(coefficients, direction)` returning RGB irradiance
- [ ] Use correct SH basis function constants from `gi_probes.py`
- [ ] Optimize: use dot products, minimize redundant normalization

**Acceptance Criteria:**
- [ ] Output matches CPU `SphericalHarmonics.evaluate()` within 1e-5
- [ ] No NaN for any input direction
- [ ] Works for degenerate directions (0,0,1) etc.
- [ ] Performance: < 20 ALU ops

---

## T-LGT-4.3: Probe Grid Interpolation Shader

**Description:** Implement trilinear interpolation between 8 corner probes.

**Tasks:**
- [ ] Implement `sample_probe_grid(world_pos, normal)` in WGSL
- [ ] Compute local coordinates from world position and grid params
- [ ] Load 8 corner probes
- [ ] Apply trilinear weights
- [ ] Handle grid boundaries (clamp or wrap)

**Acceptance Criteria:**
- [ ] Smooth interpolation visible across probe boundaries
- [ ] No discontinuities at grid edges
- [ ] Correct handling of positions outside grid
- [ ] Matches CPU `ProbeGrid.sample()` output

---

## T-LGT-4.4: DDGI Texture Creation

**Description:** Create GPU textures for DDGI irradiance and visibility.

**Tasks:**
- [ ] Create `DDGITextures` class managing irradiance and visibility textures
- [ ] Irradiance: 2D atlas, 8x8 per probe, RGBA16F
- [ ] Visibility: 2D atlas, 16x16 per probe, RG16F
- [ ] Implement `probe_to_atlas_uv(probe_index)` for texture coordinate mapping
- [ ] Add 1-texel border per probe for bilinear sampling

**Acceptance Criteria:**
- [ ] Atlas fits all probes with correct tiling
- [ ] Per-probe UV correctly addresses probe region
- [ ] Border prevents bleeding between probes
- [ ] Memory usage matches prediction

---

## T-LGT-4.5: Octahedral Encoding Shader

**Description:** Implement direction-to-octahedral and reverse mapping.

**Tasks:**
- [ ] Implement `direction_to_octahedral(direction)` returning UV in [0,1]
- [ ] Implement `octahedral_to_direction(oct_uv)` returning unit vector
- [ ] Handle hemisphere wrap correctly (z < 0)
- [ ] Add border texel handling for edge rays

**Acceptance Criteria:**
- [ ] Round-trip: `octahedral_to_direction(direction_to_octahedral(d)) == d`
- [ ] All directions map to valid UV
- [ ] All valid UV map to unit vectors
- [ ] Matches CPU `_direction_to_octahedral()` exactly

---

## T-LGT-4.6: DDGI Update Pass Interface

**Description:** Create interface for ray-traced DDGI updates.

**Tasks:**
- [ ] Create `DDGIUpdatePass` class with `trace_func` callback
- [ ] Define `DDGIRayResult` struct: `hit_distance`, `irradiance`
- [ ] Implement Fibonacci spiral ray generation matching CPU
- [ ] Write irradiance to texture (temporal blend with previous)
- [ ] Write visibility moments to texture

**Acceptance Criteria:**
- [ ] Ray directions match CPU Fibonacci spiral
- [ ] Temporal blending smooths updates (hysteresis)
- [ ] Visibility moments (mean, mean^2) computed correctly
- [ ] Works with external RT system via callback

---

## T-LGT-4.7: DDGI Prioritized Updates

**Description:** Implement per-frame probe subset updates.

**Tasks:**
- [ ] Add `DDGIProbeGrid.get_update_set(frame_index, budget)` returning probe list
- [ ] Prioritize probes near camera
- [ ] Prioritize probes with significant change
- [ ] Implement scrolling grid for large worlds
- [ ] Spread updates temporally (not all probes every frame)

**Acceptance Criteria:**
- [ ] At most `budget` probes updated per frame
- [ ] Near-camera probes update more frequently
- [ ] Changed probes converge within N frames
- [ ] Scrolling works for camera movement

---

## T-LGT-4.8: DDGI Lookup Shader

**Description:** Implement DDGI sampling with Chebyshev visibility.

**Tasks:**
- [ ] Implement `ddgi_sample(world_pos, normal)` in WGSL
- [ ] Sample irradiance from octahedral texture
- [ ] Sample visibility moments from texture
- [ ] Compute Chebyshev visibility weight
- [ ] Trilinear blend between enclosing probes

**Acceptance Criteria:**
- [ ] Indirect lighting visible from probes
- [ ] Visibility reduces light leaking through walls
- [ ] Smooth blending between probes
- [ ] Matches CPU `DDGILookup.sample()` within tolerance

---

## T-LGT-4.9: Reflection Probe Cubemap Creation

**Description:** Create cubemap array for reflection probes.

**Tasks:**
- [ ] Create `ReflectionProbeTextures` class
- [ ] Allocate cubemap array texture (configurable size, mip levels)
- [ ] Implement `capture_probe(probe_index, scene)` filling cubemap
- [ ] Generate mip chain for roughness-based lookup
- [ ] Support runtime recapture for dynamic scenes

**Acceptance Criteria:**
- [ ] Cubemap contains scene capture at probe position
- [ ] Mip chain correct for roughness mapping
- [ ] Multiple probes coexist in array
- [ ] Capture runs in separate render pass

---

## T-LGT-4.10: Parallax Box Correction Shader

**Description:** Implement reflection vector correction for finite probes.

**Tasks:**
- [ ] Implement `parallax_correct(probe, world_pos, reflection_dir)` in WGSL
- [ ] Ray-box intersection from world position
- [ ] Use intersection point for cubemap lookup direction
- [ ] Handle positions outside probe box gracefully

**Acceptance Criteria:**
- [ ] Reflections align with geometry inside probe volume
- [ ] No distortion at box corners
- [ ] Fallback to uncorrected when outside box
- [ ] Matches CPU `_parallax_correct()` output

---

## T-LGT-4.11: Reflection Probe Blending

**Description:** Blend between overlapping reflection probes.

**Tasks:**
- [ ] Implement `sample_reflection_probes(world_pos, reflection, roughness)` in WGSL
- [ ] Find probes influencing fragment (2-4 typical)
- [ ] Compute blend weights based on distance and box containment
- [ ] Sample each probe and blend
- [ ] Handle no-probe fallback (sky or ambient)

**Acceptance Criteria:**
- [ ] Smooth transitions between probe volumes
- [ ] No popping at probe boundaries
- [ ] Correct fallback outside all probes
- [ ] Performance: max 4 probe samples per fragment

---

## T-LGT-4.12: Lightmap Texture Support

**Description:** Enable baked lightmap sampling.

**Tasks:**
- [ ] Create `LightmapManager` class for texture storage
- [ ] Support per-object lightmaps or atlas
- [ ] Add UV2 (lightmap UV) to vertex format
- [ ] Implement `sample_lightmap(lightmap_index, uv2)` in shader
- [ ] Bilinear sampling with edge padding

**Acceptance Criteria:**
- [ ] Lightmapped objects show baked illumination
- [ ] No bleeding between lightmap charts
- [ ] Works with UV2 distinct from main UV
- [ ] Lightmap + real-time lighting combine correctly

---

## T-LGT-4.13: GI Integration in Lighting Shader

**Description:** Combine all GI sources in main lighting shader.

**Tasks:**
- [ ] Add `sample_indirect(world_pos, normal, roughness)` function
- [ ] Sample SH probes for diffuse indirect
- [ ] Sample DDGI for dynamic diffuse (if enabled)
- [ ] Sample reflection probes for specular indirect
- [ ] Add lightmap contribution (if UV2 present)
- [ ] Combine with direct lighting

**Acceptance Criteria:**
- [ ] Indirect lighting visible in final image
- [ ] Diffuse and specular indirect separate and correct
- [ ] Dynamic GI responds to light changes
- [ ] No double-counting between baked and dynamic

---

## T-LGT-4.14: GI Debug Visualization

**Description:** Add debug views for global illumination.

**Tasks:**
- [ ] Add SH probe visualization (colored spheres)
- [ ] Add DDGI probe grid visualization
- [ ] Add DDGI irradiance/visibility texture views
- [ ] Add reflection probe wireframe boxes
- [ ] Add lightmap-only view mode

**Acceptance Criteria:**
- [ ] Probe locations visible in debug view
- [ ] DDGI texture content inspectable
- [ ] Reflection probe volumes visible
- [ ] Lightmap contribution isolatable
