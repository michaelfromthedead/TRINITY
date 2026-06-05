# PHASE 2 TODO: Shadow Filtering Shaders

## T-LGT-2.1: PCF Filter WGSL Implementation

**Description:** Implement PCF shadow filtering in WGSL with all three sampling patterns.

**Tasks:**
- [ ] Create `shaders/shadow_pcf.wgsl` with core PCF function
- [ ] Implement `pcf_sample_grid(shadow_map, sampler, uv, depth, filter_size, samples)`
- [ ] Implement `pcf_sample_poisson(shadow_map, sampler, uv, depth, filter_size)` with precomputed disk
- [ ] Implement `pcf_sample_vogel(shadow_map, sampler, uv, depth, filter_size, samples)` with golden angle
- [ ] Add compile-time constant `PCF_PATTERN` for variant selection
- [ ] Ensure proper textureSampleCompare usage for hardware PCF

**Acceptance Criteria:**
- [ ] Grid pattern produces NxN sample shadows with correct averaging
- [ ] Poisson disk produces irregular soft shadows without banding
- [ ] Vogel disk produces smooth circular soft shadows
- [ ] Performance matches CPU reference implementation
- [ ] Visual comparison test passes against reference renders

---

## T-LGT-2.2: PCSS Filter WGSL Implementation

**Description:** Implement PCSS with blocker search and variable penumbra.

**Tasks:**
- [ ] Create `shaders/shadow_pcss.wgsl` with PCSS algorithm
- [ ] Implement `pcss_blocker_search(shadow_map, uv, receiver_depth, search_radius)` returning average blocker depth
- [ ] Implement `estimate_penumbra(receiver_depth, blocker_depth, light_size)` matching CPU formula
- [ ] Implement `pcss_filter(shadow_map, sampler, uv, depth, penumbra_size)` calling PCF with variable width
- [ ] Add early-out when no blockers found (return 1.0)
- [ ] Add early-out when fully occluded (return 0.0)

**Acceptance Criteria:**
- [ ] Penumbra widens with distance from occluder
- [ ] Close shadows remain sharp, distant shadows become soft
- [ ] Blocker search radius configurable via uniform
- [ ] Performance acceptable (< 2ms for 32 search + 32 filter samples)

---

## T-LGT-2.3: VSM Filter WGSL Implementation

**Description:** Implement Variance Shadow Maps with Chebyshev test.

**Tasks:**
- [ ] Create `shaders/shadow_vsm.wgsl` with VSM algorithm
- [ ] Implement `vsm_visibility(moments, depth, min_variance, bleed_reduction)`
- [ ] Use Chebyshev's inequality for soft shadow computation
- [ ] Implement light bleeding reduction via linear clamping
- [ ] Add configurable minimum variance for numerical stability

**Acceptance Criteria:**
- [ ] Soft shadows without PCF sampling cost
- [ ] Light bleeding controlled by `bleed_reduction` parameter (0.0-0.3)
- [ ] No NaN or Inf values from variance computation
- [ ] Matches CPU `VSMFilter` output for test cases

---

## T-LGT-2.4: ESM Filter WGSL Implementation

**Description:** Implement Exponential Shadow Maps.

**Tasks:**
- [ ] Create `shaders/shadow_esm.wgsl` with ESM algorithm
- [ ] Implement `esm_visibility(exp_depth, receiver_depth, exponent)`
- [ ] Handle overflow with clamped exponent range (40-80)
- [ ] Implement pre-filtered ESM blur pass if needed

**Acceptance Criteria:**
- [ ] Smooth soft shadows with single texture sample
- [ ] No overflow artifacts at exponent=80
- [ ] Works with R32Float shadow map format
- [ ] Performance: single texture sample + exp operations

---

## T-LGT-2.5: Contact Shadow Ray March

**Description:** Implement screen-space ray marching for contact shadows.

**Tasks:**
- [ ] Create `shaders/shadow_contact.wgsl` with ray march algorithm
- [ ] Implement `contact_shadow(depth_buffer, screen_pos, light_dir_ss, max_steps, thickness)`
- [ ] Add jittered ray start for temporal stability
- [ ] Implement thickness-aware occlusion test
- [ ] Add early-out when ray exits screen bounds

**Acceptance Criteria:**
- [ ] Shadows appear at object contact points
- [ ] No false occlusions from front faces
- [ ] Configurable step count (16-64) via uniform
- [ ] Temporal stability with jitter + TAA

---

## T-LGT-2.6: Shadow Filter Uniform Buffer

**Description:** Create GPU uniform buffer for shadow filter parameters.

**Tasks:**
- [ ] Define `ShadowFilterUniforms` struct in WGSL matching Python config
- [ ] Add `filter_size`, `pcss_light_size`, `vsm_min_variance`, `vsm_bleed_reduction`
- [ ] Add `esm_exponent`, `contact_max_steps`, `contact_thickness`
- [ ] Implement Python-side buffer packing with 16-byte alignment
- [ ] Add per-light filter override support

**Acceptance Criteria:**
- [ ] Uniform buffer uploads without alignment errors
- [ ] Shader reads all parameters correctly
- [ ] Per-light overrides work (e.g., important lights get more samples)
- [ ] Hot-reload of filter parameters for tuning

---

## T-LGT-2.7: Shadow Filter Shader Permutations

**Description:** Implement shader permutation system for filter variants.

**Tasks:**
- [ ] Create `ShadowFilterPermutation` enum: PCF, PCSS, VSM, ESM
- [ ] Create `PCFPatternPermutation` enum: GRID, POISSON, VOGEL
- [ ] Implement shader preprocessor with `#define` injection
- [ ] Cache compiled shader variants by permutation key
- [ ] Add fallback to simpler filter if compile fails

**Acceptance Criteria:**
- [ ] Only requested permutations compiled on demand
- [ ] Shader cache persists across sessions
- [ ] Invalid permutations (e.g., ESM + POISSON) rejected at compile time
- [ ] Total compiled variants < 25

---

## T-LGT-2.8: Integration with Main Lighting Shader

**Description:** Wire shadow filtering into the deferred/forward lighting shader.

**Tasks:**
- [ ] Add `sample_shadow(light_index, world_pos, normal)` function to lighting shader
- [ ] Select filter function based on light's `shadow_filter_type`
- [ ] Transform world position to shadow UV using light's view-proj
- [ ] Apply normal bias before shadow lookup
- [ ] Combine shadow map + contact shadow visibility

**Acceptance Criteria:**
- [ ] Shadows appear in final rendered image
- [ ] Filter type selectable per light
- [ ] No visible seams at cascade boundaries (blend zone)
- [ ] Contact shadows add detail without doubling shadowing

---

## T-LGT-2.9: Shadow Filter Debug Visualization

**Description:** Add debug views for shadow filter inspection.

**Tasks:**
- [ ] Add penumbra size visualization for PCSS
- [ ] Add blocker count visualization for PCSS
- [ ] Add variance visualization for VSM
- [ ] Add contact shadow ray visualization
- [ ] Wire to debug UI toggle

**Acceptance Criteria:**
- [ ] PCSS penumbra size shown as color gradient
- [ ] VSM variance shown as intensity
- [ ] Contact shadow rays drawn as lines when enabled
- [ ] No performance impact when debug disabled
