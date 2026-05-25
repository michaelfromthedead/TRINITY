# PHASE 3 TODO: Clustered Lighting

## T-LGT-3.1: Froxel Grid GPU Buffer

**Description:** Upload froxel grid bounds to GPU storage buffer.

**Tasks:**
- [ ] Create `FroxelGPU` struct with AABB min/max (32 bytes per froxel)
- [ ] Add `FroxelGrid.to_gpu_buffer()` method returning packed bytes
- [ ] Create storage buffer via renderer-backend with correct size
- [ ] Upload on camera change (FOV, near, far, or position significant change)
- [ ] Add WGSL struct definition matching Python packing

**Acceptance Criteria:**
- [ ] Buffer contains X*Y*Z froxels with correct AABB bounds
- [ ] WGSL shader reads froxel bounds correctly
- [ ] Re-upload only when camera parameters change
- [ ] Unit test validates buffer contents against CPU computation

---

## T-LGT-3.2: Light List Buffer Packing

**Description:** Pack per-froxel light lists into GPU-uploadable format.

**Tasks:**
- [ ] Create `LightListPacker` class for buffer packing
- [ ] Implement `pack_headers(light_lists)` returning array of (offset, count) pairs
- [ ] Implement `pack_indices(light_lists)` returning flattened index array
- [ ] Handle froxels with zero lights (count=0)
- [ ] Add overflow check when total indices exceed buffer capacity

**Acceptance Criteria:**
- [ ] Headers contain correct offset into indices array
- [ ] Counts match actual number of lights per froxel
- [ ] Empty froxels have count=0, offset=arbitrary
- [ ] Overflow raises clear error before GPU corruption

---

## T-LGT-3.3: Light Data Buffer

**Description:** Create GPU buffer containing all light data.

**Tasks:**
- [ ] Define `LightGPU` struct (64 bytes) with all light properties
- [ ] Add `LightManager.to_gpu_buffer()` packing all active lights
- [ ] Include light type, position, direction, color, intensity, attenuation params
- [ ] Include shadow index (-1 if no shadow)
- [ ] Handle special cases: IES profile index, area light dimensions

**Acceptance Criteria:**
- [ ] Buffer contains all active scene lights
- [ ] Light indices in froxel lists correspond to correct lights
- [ ] Shadow index correctly references shadow map array
- [ ] Unit test validates round-trip (Python -> GPU -> Python comparison)

---

## T-LGT-3.4: Light Evaluation Shader

**Description:** Implement per-light-type evaluation functions in WGSL.

**Tasks:**
- [ ] Create `shaders/light_eval.wgsl` with light evaluation functions
- [ ] Implement `evaluate_directional(light, normal)` with simple NdotL
- [ ] Implement `evaluate_point(light, pos, normal)` with smooth distance falloff
- [ ] Implement `evaluate_spot(light, pos, normal)` with angular attenuation
- [ ] Add switch statement dispatcher based on `light.light_type`

**Acceptance Criteria:**
- [ ] Directional light produces correct parallel lighting
- [ ] Point light falloff matches inverse-square with smooth cutoff
- [ ] Spot light has correct inner/outer angle transition
- [ ] No division by zero or NaN in edge cases

---

## T-LGT-3.5: Smooth Attenuation Functions

**Description:** Implement physically-plausible smooth attenuation.

**Tasks:**
- [ ] Implement `smooth_distance_attenuation(dist, radius)` matching CPU formula
- [ ] Implement `smooth_angular_attenuation(angle, inner, outer)` for spots
- [ ] Ensure attenuation reaches exactly 0.0 at radius/outer_angle
- [ ] Ensure attenuation reaches exactly 1.0 at zero dist / inner_angle

**Acceptance Criteria:**
- [ ] No harsh cutoff visible at light radius
- [ ] Spot light edges are smooth, not sharp
- [ ] Attenuation formula matches `light_types.py` reference
- [ ] Performance: minimal ALU per light

---

## T-LGT-3.6: Froxel Index Computation

**Description:** Compute froxel index from world position in shader.

**Tasks:**
- [ ] Implement `compute_froxel_index(world_pos)` in WGSL
- [ ] Project world position to screen space
- [ ] Compute XY cell from screen coordinates
- [ ] Compute Z slice from linearized depth using exponential formula
- [ ] Clamp to grid bounds

**Acceptance Criteria:**
- [ ] Index matches CPU `FroxelGrid.get_froxel_index()`
- [ ] Works at screen edges and corners
- [ ] Works at near plane and far plane
- [ ] No out-of-bounds access

---

## T-LGT-3.7: Clustered Lighting Loop

**Description:** Implement per-fragment clustered lighting loop.

**Tasks:**
- [ ] Add clustered lighting to main fragment shader
- [ ] Read froxel index from world position
- [ ] Load light list header (offset, count)
- [ ] Iterate `count` lights starting at `offset`
- [ ] Accumulate radiance from all lights
- [ ] Apply shadow visibility per light (if shadow_index >= 0)

**Acceptance Criteria:**
- [ ] Fragment lit by correct lights based on froxel membership
- [ ] No lights missed that should affect fragment
- [ ] No lights evaluated that don't affect fragment
- [ ] Performance scales with actual visible lights, not total lights

---

## T-LGT-3.8: Light Buffer Upload Path

**Description:** Wire light buffers to renderer-backend upload system.

**Tasks:**
- [ ] Add `LightingSystem.upload_buffers(gpu_context)` method
- [ ] Upload froxel buffer if camera changed
- [ ] Upload light list headers and indices every frame
- [ ] Upload light data buffer every frame
- [ ] Use staging buffers for large uploads

**Acceptance Criteria:**
- [ ] Buffers uploaded before lighting shader executes
- [ ] No GPU stalls from upload/render overlap
- [ ] Frame graph respects buffer upload dependencies
- [ ] Memory usage stable (no per-frame allocation)

---

## T-LGT-3.9: IES Profile Support

**Description:** Enable IES light profile evaluation in shader.

**Tasks:**
- [ ] Create IES profile texture array (1D textures per profile)
- [ ] Upload IES data from `IESLight.profile` to texture
- [ ] Implement `sample_ies_profile(profile_index, angle)` in WGSL
- [ ] Multiply IES intensity into point light evaluation

**Acceptance Criteria:**
- [ ] IES lights show characteristic photometric patterns
- [ ] Multiple IES profiles supported simultaneously
- [ ] Bilinear sampling matches CPU reference
- [ ] IES fallback to uniform when profile unavailable

---

## T-LGT-3.10: Clustered Lighting Debug Visualization

**Description:** Add debug views for clustered lighting inspection.

**Tasks:**
- [ ] Add froxel grid visualization (wireframe boxes)
- [ ] Add heatmap of lights-per-froxel
- [ ] Add light influence visualization (which froxels light affects)
- [ ] Wire to debug UI toggle

**Acceptance Criteria:**
- [ ] Froxel grid shows 3D structure in debug view
- [ ] Heatmap reveals hotspots with many lights
- [ ] Individual light influence visible when selected
- [ ] No performance impact when debug disabled
