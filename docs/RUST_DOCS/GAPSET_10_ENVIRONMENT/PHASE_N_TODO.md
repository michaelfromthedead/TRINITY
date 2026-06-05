# GAPSET_10_ENVIRONMENT -- Task Checklist

> **Task ID Format**: T-ENV-{PHASE}.{N}
> **Total Tasks**: 38
> **Phase 1**: 13 tasks (T-ENV-1.1 through T-ENV-1.13)
> **Phase 2**: 13 tasks (T-ENV-2.1 through T-ENV-2.13)
> **Phase 3**: 12 tasks (T-ENV-3.1 through T-ENV-3.12)

---

## Phase 1: Atmosphere Foundation & Water/Terrain Core (13 tasks)

### T-ENV-1.1: Bruneton LUT Precomputation
- **Gaps**: S11-G3
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: S15 math.rs (Vec3, Mat4), S16 asset pipeline (LUT storage)
- **Acceptance Criteria**:
  - [ ] CPU precompute Transmittance LUT at 256x64 resolution, RGBA16F format
  - [ ] CPU precompute Sky-View LUT at 256x512 resolution, RGB16F format
  - [ ] CPU precompute Aerial Perspective LUT at 32x32x32 resolution (inscatter + transmittance)
  - [ ] Rayleigh + Mie scattering coefficients computed from altitude
  - [ ] Cornette-Shanks Mie phase function with configurable asymmetry g
  - [ ] Optional ozone absorption (Chappuis band)
  - [ ] Multi-scattering approximation (single scatter + energy compensation)
  - [ ] LUT plausibility test: non-zero at all angles, max >0.5 near sun
  - [ ] Precompute time under 500ms on CPU (Python+ndarray reference)
  - [ ] Deterministic output: same sun angle + turbidity = identical LUT

### T-ENV-1.2: Sky Rendering Pass
- **Gaps**: S11-G2
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-1.1 (LUTs must precompute), S1 frame graph (pass declaration)
- **Acceptance Criteria**:
  - [ ] Full-screen triangle pass declared in S1 frame graph
  - [ ] Vertex shader: pass-through, compute world-space view direction
  - [ ] Pixel shader: sample Sky-View LUT by (view_zenith_cos, view_azimuth)
  - [ ] Depth test with greater_equal, write sky only where no geometry
  - [ ] HDR output (RGB16F render target)
  - [ ] Works at all sun angles (noon, sunset, twilight, night)
  - [ ] No color banding (16-bit LUT precision sufficient)
  - [ ] Aerial perspective integration (LUT 3 sampled per pixel with depth)

### T-ENV-1.3: Sun Disk, Moon & Star Field
- **Gaps**: S11-G9
- **Effort**: LOW (2-3 days)
- **Dependencies**: T-ENV-1.2 (sky pass exists)
- **Acceptance Criteria**:
  - [ ] Sun disk rendering with configurable angular radius (~0.267 degrees default)
  - [ ] Sun brightness modulated by atmospheric transmittance along sun direction
  - [ ] Sun glow: additive bloom-friendly radial gradient beyond disk
  - [ ] Moon rendering with lunar phase (lit fraction from sun-moon-earth angle)
  - [ ] Moon surface: albedo texture or procedural crater pattern
  - [ ] Moon brightess: diffuse reflector model (sun_radiance * albedo * phase_factor)
  - [ ] Star field: 2,000-8,000 procedural stars with magnitude distribution
  - [ ] Star twinkle: sin(time * frequency + phase) modulation
  - [ ] Milky Way: procedural noise band across sky dome
  - [ ] No visible tiling or artifacts in star distribution

### T-ENV-1.4: Froxel Volume Management
- **Gaps**: S11-G1
- **Effort**: MEDIUM (4-6 days)
- **Dependencies**: S1 frame graph, S15 math.rs
- **Acceptance Criteria**:
  - [ ] @component FroxelConfig defined (grid dimensions, depth partitions)
  - [ ] Froxel 3D texture allocation: radiance + extinction (RGBA16F x2)
  - [ ] Logarithmic depth partitioning with configurable near/far planes
  - [ ] Base resolution: 64x48x32 (configurable: 32x24x16 to 256x192x128)
  - [ ] Froxel world-space coordinate computation from (x, y, depth_slice)
  - [ ] Quality tier enum: Low/Medium/High/Ultra with resolution presets
  - [ ] Froxel volume bound as UAV for compute shader output
  - [ ] Froxel volume bound as SRV for compositing shader input

### T-ENV-1.5: Froxel Density Field & Light Scattering
- **Gaps**: S11-G2
- **Effort**: HIGH (5-8 days)
- **Dependencies**: T-ENV-1.4 (froxel volume), S4 lighting (sun direction + intensity)
- **Acceptance Criteria**:
  - [ ] Compute shader dispatch: one thread per froxel
  - [ ] Uniform fog density with height falloff (exponential, configurable height + rate)
  - [ ] Layered fog: 2+ independent density layers at different altitudes
  - [ ] Local fog volumes: artist-placed box/sphere with @component FogVolume
  - [ ] Sun scattering: sun_radiance * transmittance_sun * phase_function(dot(view, sun))
  - [ ] Henyey-Greenstein phase function with configurable g
  - [ ] Froxel outputs: radiance (RGB) + extinction (RGB) per cell
  - [ ] Transmittance computed: exp(-extinction * step_size) along view ray
  - [ ] Energy conservation: incident = transmitted + absorbed (+/- 0.01)
  - [ ] Optional: density debug visualization

### T-ENV-1.6: Froxel Compositing
- **Gaps**: S11-G2
- **Effort**: MEDIUM (3-4 days)
- **Dependencies**: T-ENV-1.5 (froxel scattering)
- **Acceptance Criteria**:
  - [ ] Full-screen composite pass (Approach B: single froxel lookup per pixel)
  - [ ] Per-pixel: sample froxel volume at (x, y, depth), apply radiance + extinction
  - [ ] Output: final_color = scene_color * transmittance + radiance
  - [ ] Optional high-quality mode (Approach A: ray march through froxels)
  - [ ] Depth-aware: froxel lookup matches pixel depth
  - [ ] No visible froxel boundaries (bilinear interpolation in XY, nearest in Z)

### T-ENV-1.7: Gerstner Wave Compute Shader
- **Gaps**: S12-G1
- **Effort**: MEDIUM (4-6 days)
- **Dependencies**: S1 frame graph, S15 math.rs
- **Acceptance Criteria**:
  - [ ] @component GerstnerWaveSet: num_waves (8-32), amplitudes, wavelengths, directions, steepness
  - [ ] Compute shader evaluating N waves per vertex
  - [ ] Vertical displacement: SUM(A_i * sin(w_i * dot(D_i, (x,z)) + t * phase_i))
  - [ ] Horizontal displacement (choppy): SUM(Q_i * A_i * D_i * cos(...))
  - [ ] Steepness control: Q_i = 1/(w_i * A_i * numWaves) prevents loops
  - [ ] Analytic normal calculation from displacement gradient (not finite difference)
  - [ ] Deep water dispersion relation: w^2 = g * k
  - [ ] Wave distribution: 1-2 large swells, 4-8 medium chop, 4-8 small detail
  - [ ] Wave parameters bound as uniform buffer (16 waves x 4 floats)
  - [ ] Performance: <0.1ms for 256x256 grid with 16 waves
  - [ ] Quality tier: 8/16/24/32 waves for Low/Medium/High/Ultra

### T-ENV-1.8: Water Shading Pass
- **Gaps**: S12-G2
- **Effort**: HIGH (6-10 days)
- **Dependencies**: T-ENV-1.7 (Gerstner displacement), S4 lighting, S7 reflections
- **Acceptance Criteria**:
  - [ ] @component WaterShadingConfig: Fresnel, refraction, reflection, specular params
  - [ ] Fresnel effect: Schlick approximation with F0=0.02 (IOR ~1.33)
  - [ ] Fresnel at 0 degrees = F0, at 90 degrees = 1.0 (verifiable)
  - [ ] Reflection: SSR (screen-space ray march, HZB-accelerated)
  - [ ] Reflection: Planar reflection (rendered to separate RT, clip above water)
  - [ ] Reflection: Probe fallback (cubemap at water surface location)
  - [ ] Reflection blend: SSR near, probe far, planar for local planar surfaces
  - [ ] Refraction: scene behind water, offset by surface normal distortion
  - [ ] Refraction strength: configurable (typical max 0.05 screen-space)
  - [ ] Specular: GGX with anisotropic stretching (stretched along wave direction)
  - [ ] Subsurface scattering for shallow water (depth-dependent color blend)
  - [ ] Shallow color / deep color interpolation by depth
  - [ ] Water color integration with sky reflection (sky color drives water appearance)
  - [ ] Performance: <0.3ms at 1080p

### T-ENV-1.9: Terrain Clipmap Compute Shader
- **Gaps**: S12-G3
- **Effort**: HIGH (6-10 days)
- **Dependencies**: S1 frame graph, S15 math.rs
- **Acceptance Criteria**:
  - [ ] @component ClipmapConfig: grid_size (128), num_levels (8), finest_spacing (0.5m)
  - [ ] Nested regular grids centered on camera, constant vertex count per level
  - [ ] Level 0: 128x128 vertices, 0.5m spacing, 64m coverage
  - [ ] Level N: 128x128 vertices, 2^N * 0.5m spacing, 64 * 2^N m coverage
  - [ ] Ring buffer update: shift grid when camera moves >1 cell, upload new strip
  - [ ] Compute shader: heightfield texture -> vertex buffer (position + normal)
  - [ ] Normal calculation: central differencing on heightfield (finite difference)
  - [ ] Geomorphing between levels: lerp(height_N, height_N+1, morph_alpha)
  - [ ] Morph alpha: saturate((edge_distance - morph_start * half_size) / ((1-morph_start) * half_size))
  - [ ] Indirect draw call generation per clipmap level
  - [ ] Multi-draw-indirect for all levels in one draw call
  - [ ] Total clipmap overhead: <0.4ms per frame (all 8 levels)
  - [ ] Height data upload per strip: <0.05ms
  - [ ] No visible popping at any camera speed (geomorphing handles transitions)

### T-ENV-1.10: Terrain Material Blending
- **Gaps**: S12-G4
- **Effort**: HIGH (5-8 days)
- **Dependencies**: T-ENV-1.9 (clipmap provides vertex positions)
- **Acceptance Criteria**:
  - [ ] @component TerrainMaterialConfig: splat_resolution (2048), max_layers (8)
  - [ ] @component TerrainLayerDef: albedo, normal, mask textures + uv_scale
  - [ ] RGBA splat maps: 4 layers per texture, 2 textures for 8 total layers
  - [ ] Weight normalization: weights /= max(dot(weights, 1.0), 0.001)
  - [ ] Slope-based auto-blend: smoothstep(slope_min, slope_max, slope)
  - [ ] Height-based auto-blend: smoothstep(height_min, height_max, world_pos.y)
  - [ ] Curvature-based blending: convex/concave detection for path/river materials
  - [ ] Stochastic sampling: random UV offset per-pixel to reduce tiling
  - [ ] Bindless texture arrays for material layers (RGBA splat + N x albedo/normal/mask)
  - [ ] Output: albedo, normal, roughness, metallic, ambient occlusion per pixel

### T-ENV-1.11: Foliage GPU Instancing
- **Gaps**: S12-G5
- **Effort**: HIGH (6-10 days)
- **Dependencies**: S1 frame graph, S2 GPU-driven culling
- **Acceptance Criteria**:
  - [ ] @component FoliageInstanceBuffer: max_instances (1,000,000), stride (32 bytes)
  - [ ] @foliage_type: density, cull_distance, collision, wind_response params
  - [ ] @lod: 3 levels (mesh, cross-plane, billboard) with distance thresholds
  - [ ] Instance data: pos (float3), scale (float), rotation (float quaternion packed), color (RGBA8)
  - [ ] GPU frustum culling compute shader (AABB test against 6 frustum planes)
  - [ ] GPU distance/LOD culling (select mesh/cross-plane/billboard per instance)
  - [ ] Compact survivor list via GPU prefix sum or atomic counter
  - [ ] Multi-draw-indirect from compacted list
  - [ ] LOD cross-fade: alpha blend between LOD levels over 10m transition zone
  - [ ] Billboard: camera-facing quad with alpha texture for far instances
  - [ ] Cross-plane: 2-3 intersecting quads for medium distance
  - [ ] Full mesh: 100% vertex count for near instances
  - [ ] Grass: 500K max instances, no shadow casting
  - [ ] Shrubs: 100K max instances
  - [ ] Trees: 50K max instances, shadow casting (dynamic for trees)
  - [ ] GPU culling time: <0.3ms for 100K instances

### T-ENV-1.12: Create 6 Missing Engine Directories
- **Gaps**: S12-G6
- **Effort**: LOW (1-2 days)
- **Dependencies**: None (pure directory + stub creation)
- **Acceptance Criteria**:
  - [ ] `engine/rendering/terrain/` created with terrain_lod.py, terrain_material.py, foliage.py stubs
  - [ ] `engine/rendering/water/` created with water_simulation.py, water_shading.py, foam.py stubs
  - [ ] `engine/rendering/texturing/` created with virtual_texturing.py, texture_compression.py stubs
  - [ ] `engine/world/partition/` created with grid.py, streaming.py stubs
  - [ ] `engine/world/terrain/` created with patch.py, heightfield.py stubs
  - [ ] `engine/world/foliage/` created with placement.py, biome.py stubs
  - [ ] Each stub file contains @component class definitions matching spec
  - [ ] Stub classes import from Trinity metaclass/decorator base classes
  - [ ] All stubs pass Python import validation (no syntax errors)

### T-ENV-1.13: Create 9 Missing Trinity Decorators
- **Gaps**: S12-G7
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: Trinity decorator base classes (tier system, metaclass protocol)
- **Acceptance Criteria**:
  - [ ] @terrain_patch defined with size, overlap, height_data params
  - [ ] @heightfield defined with resolution, height_scale, height_bias params
  - [ ] @terrain_layer defined with index, name, blend_mode params
  - [ ] @grass_type defined with density, blade_height, color_variation params
  - [ ] @scatter_rule defined with noise_type, density, slope_range, height_range params
  - [ ] @biome defined with climate_zone, vegetation_set, terrain_materials params
  - [ ] @weather_zone defined with coverage_range, wind_range, fog_range params
  - [ ] @hlod_layer defined with level, cell_size, merge_threshold params
  - [ ] @environment_volume defined with shape, blend_radius, priority params
  - [ ] Each decorator uses proper tier numbering (Tier 48 for world building)
  - [ ] Each decorator integrates with Registry for introspection

---

## Phase 2: Clouds, FFT Ocean & Virtual Texturing (13 tasks)

### T-ENV-2.1: Cloud Noise Textures
- **Gaps**: S11-G4
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: None (can be baked offline or generated at init)
- **Acceptance Criteria**:
  - [ ] 3D Worley noise texture at 32^3 resolution, RG8 format (2 octaves)
  - [ ] Detail noise at 16^3 resolution, R8 format
  - [ ] Base shape noise: Perlin-Worley FBM (4-5 octaves)
  - [ ] Detail noise: Worley FBM (2-3 octaves) for wispy edges
  - [ ] Tiling: 4-8km tile size with cross-fade at boundaries
  - [ ] Texel animation: detail noise scrolls at different rate than base
  - [ ] Option A: CPU generation at init (portable)
  - [ ] Option B: Bake via S16 asset pipeline (production)
  - [ ] Memory: <2MB for all noise textures combined

### T-ENV-2.2: Cloud Ray Marching Pass
- **Gaps**: S11-G2
- **Effort**: HIGH (6-10 days)
- **Dependencies**: T-ENV-2.1 (noise textures), T-ENV-1.2 (sky LUT for cloud lighting)
- **Acceptance Criteria**:
  - [ ] Full-screen compute pass: one thread per pixel, trace ray through cloud layer
  - [ ] Cloud layer defined as horizontal slab: min_height (1-8km), thickness (2-8km)
  - [ ] Density remapping: shifted raw_noise by coverage, eroded by detail noise
  - [ ] Coverage from 2D weather map (temporary: uniform float until T-ENV-3.1)
  - [ ] Cloud type: cumulus vs. stratus distinction from weather map
  - [ ] Step count: 64-128 (configurable per quality tier)
  - [ ] Early termination: break when transmittance < 0.01
  - [ ] Depth-aware: stop marching behind solid geometry
  - [ ] Adaptive steps: fewer steps for distant clouds (>500m = 48 steps)
  - [ ] Beer's law: segment_transmittance = exp(-density * step_size * extinction)
  - [ ] Powder effect: 1.0 - exp(-density * powder_factor) for bright edges
  - [ ] Multi-scattering approximation: single_scatter / max(1 - albedo * (1 - transmittance), 0.001)
  - [ ] Ambient light: blend sky color (bottom) with sun color (top) by cloud height
  - [ ] Quality tiers: Low 32, Medium 64, High 128, Ultra 256 steps

### T-ENV-2.3: Cloud Lighting (Beer + Powder + Multi-Scatter)
- **Gaps**: S11-G10, S11-G11
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-2.2 (ray marching pass)
- **Acceptance Criteria**:
  - [ ] Beer's law extinction: transmittance = exp(-density * distance) verified against reference
  - [ ] Powder effect brightening: verified at cloud boundaries (density -> 0)
  - [ ] Multi-scattering energy conservation: forward_scatter = single / (1 - albedo * (1 - transmittance))
  - [ ] Multi-scatter produces brighter cloud interiors than single-scatter only
  - [ ] No energy gain: output radiance <= input radiance
  - [ ] Configurable extinction coefficient, powder_factor, albedo

### T-ENV-2.4: Cloud Shadows on Terrain
- **Gaps**: S11-G12
- **Effort**: MEDIUM (4-6 days)
- **Dependencies**: T-ENV-2.2 (cloud density field), T-ENV-1.9 (terrain clipmap)
- **Acceptance Criteria**:
  - [ ] Analytic ray trace per terrain shading point: 16 steps through cloud layer toward sun
  - [ ] Density accumulation along ray: step through cloud layer only (2-8km)
  - [ ] Shadow factor: exp(-density_sum * shadow_extinction)
  - [ ] Temporal accumulation: blend 0.9 previous, 0.1 current for stability
  - [ ] Integrated into terrain lighting: sun_irradiance *= cloud_shadow_factor
  - [ ] Shadow moves with cloud drift (wind direction + speed)
  - [ ] No visible flickering at 60fps with camera movement
  - [ ] Performance: <0.3ms at 1080p

### T-ENV-2.5: God Rays (Volumetric Light Shafts)
- **Gaps**: S11-G5
- **Effort**: MEDIUM (4-6 days)
- **Dependencies**: T-ENV-1.5 (froxel scattering), T-ENV-2.2 (cloud rendering)
- **Acceptance Criteria**:
  - [ ] Volumetric base: sun scattering through froxels (reuse froxel scattering from T-ENV-1.5)
  - [ ] Sun transmittance computed per froxel: ray march toward sun, accumulate density
  - [ ] Screen-space detail: radial blur from sun screen position on visibility buffer
  - [ ] Visibility buffer: 0=shadowed, 1=lit (from shadow maps or cloud shadow)
  - [ ] Radial samples: 32/64 for Medium/High quality
  - [ ] Hybrid blend: lerp(volumetric, screen_space, detail_blend_factor)
  - [ ] Composited additive: final_color += godray_color * intensity
  - [ ] Sun must be on-screen for screen-space component (volumetric always active)

### T-ENV-2.6: Temporal Reprojection for Fog and Clouds
- **Gaps**: S11-G7
- **Effort**: HIGH (5-8 days)
- **Dependencies**: T-ENV-1.5 (froxel), T-ENV-2.2 (clouds)
- **Acceptance Criteria**:
  - [ ] Velocity buffer: screen-space motion vectors from camera movement
  - [ ] Froxel reprojection: reproject previous frame froxels using previous camera matrices
  - [ ] Cloud reprojection: reproject previous frame cloud buffer
  - [ ] Blend factor: result = lerp(previous_reprojected, current, 0.1)
  - [ ] Rejection on depth discontinuity (depth delta > threshold)
  - [ ] Rejection on radiance change (large change = no history)
  - [ ] Full reset on camera cut or teleport
  - [ ] Upscale: bilateral filter from half-res to full-res
  - [ ] Temporal stability: static camera = stable image within 10 frames
  - [ ] Quality saving: 2-4x effective resolution at similar cost

### T-ENV-2.7: LUT Cooking Pipeline (S16 Integration)
- **Gaps**: S11-G8
- **Effort**: LOW (2-3 days)
- **Dependencies**: T-ENV-1.1 (LUT precompute), S16 asset pipeline
- **Acceptance Criteria**:
  - [ ] Cooked LUT stored as .trinity_lut asset format (header + RGBA16F data)
  - [ ] LUT loading from disk at engine init (skip CPU recomputation)
  - [ ] LUT versioning: recompute when sun model parameters change
  - [ ] Fallback: CPU recompute if cooked LUT not found

### T-ENV-2.8: FFT Ocean Compute Shader
- **Gaps**: S12-G9
- **Effort**: HIGH (6-10 days)
- **Dependencies**: S1 frame graph, S15 math.rs (complex numbers)
- **Acceptance Criteria**:
  - [ ] @component FFTOceanConfig: fft_size (256), patch_size (500m), wind_speed, direction
  - [ ] CPU: Generate Phillips spectrum h0(K) with Gaussian noise
  - [ ] GPU: Time evolution: h(K,t) = h0(K)*exp(i*w*t) + h0*(-K)*exp(-i*w*t)
  - [ ] IFFT row pass: 1D IFFT along X for each row
  - [ ] IFFT column pass: 1D IFFT along Z for each column
  - [ ] Output: heightfield texture (R32F)
  - [ ] Optional: slope field output for normal computation
  - [ ] Optional: displacement field (choppy waves via gradient of heightfield)
  - [ ] Single cascade: covers 500m patch at 256x256 resolution
  - [ ] Phillips constant, chop_amount, wind parameters as uniform buffer
  - [ ] Verification: IFFT(FFT(x)) == x within floating point precision
  - [ ] Performance: <0.3ms for 256x256 FFT

### T-ENV-2.9: Foam Generation (Crest + Shore)
- **Gaps**: S12-G2
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-1.7 (Gerstner), T-ENV-2.8 (FFT for crest foam)
- **Acceptance Criteria**:
  - [ ] @component FoamConfig: crest_threshold (0.7), decay_rate (2.0), shore_width (5.0)
  - [ ] Crest foam from Jacobian: J < 0 folded wave -> foam = clamp(1 - J/threshold, 0, 1)
  - [ ] Crest foam decay: foam(t) = max(0, foam - decay_rate * dt)
  - [ ] Shore foam from shoreline distance: band width configurable
  - [ ] Shore foam combined with wave height for breaking wave foam
  - [ ] Foam mask render target (full-res, R8 format)
  - [ ] Composited on water surface as additive overlay with foam texture
  - [ ] Foam texture: procedural noise pattern for detail
  - [ ] Simulated foam: placeholder for Phase 3 advection-based system

### T-ENV-2.10: Virtual Texturing -- Page Table
- **Gaps**: S12-G8
- **Effort**: HIGH (5-8 days)
- **Dependencies**: S1 frame graph, S14 RHI (bindless textures, storage buffers)
- **Acceptance Criteria**:
  - [ ] @component VirtualTextureConfig: virtual_size (131072), tile_size (128), page_table_size (1024)
  - [ ] Page table: 1024x1024 RGBA16Uint texture (8MB)
  - [ ] Page table entry: physical_x (u16), physical_y (u16), mip_level (u8), flags (u8)
  - [ ] Flag bit 0: resident, bit 1: requested, bit 2: streaming, bit 3: invalid
  - [ ] Shader sampling function: sample_virtual_texture(vt_uv, layer) -> vec4f
  - [ ] Non-resident pages return fallback_color (configurable)
  - [ ] Page table update: GPU write or CPU via staging buffer
  - [ ] 128K x 128K virtual address space = 1024 x 1024 virtual tiles

### T-ENV-2.11: Virtual Texturing -- Physical Atlas
- **Gaps**: S12-G8
- **Effort**: HIGH (5-8 days)
- **Dependencies**: T-ENV-2.10 (page table)
- **Acceptance Criteria**:
  - [ ] Physical atlas: 16384 x 16384 texels (16K x 16K = 256M texels)
  - [ ] Atlas divided into (16384/128)^2 = 128^2 = 16,384 tile slots
  - [ ] Each tile in atlas includes border texels (2 texel padding per side)
  - [ ] Actual tile storage: (128+4) x (128+4) = 132 x 132 per tile
  - [ ] Mip chain: log2(16384/4) = 13 mip levels, each stored as separate page table
  - [ ] Physical atlas as texture_2d_array (1 layer per mip or separate atlases)
  - [ ] LRU eviction queue: free list + deque for tile slot management
  - [ ] Allocate/free tile operations with O(1) amortized cost
  - [ ] Upload tile data to GPU: staging buffer -> atlas copy
  - [ ] Evicted tiles: flush if dirty (page has been modified)

### T-ENV-2.12: Virtual Texturing -- Feedback Pass
- **Gaps**: S12-G8
- **Effort**: HIGH (4-7 days)
- **Dependencies**: T-ENV-2.10 (page table), T-ENV-1.9 (terrain uses VT)
- **Acceptance Criteria**:
  - [ ] Feedback pass: render virtual UV coordinates to feedback buffer
  - [ ] Feedback buffer: 1024x1024 R32Uint texture
  - [ ] For each visible pixel: write encoded (virtual_tile_x, virtual_tile_y, mip) to buffer
  - [ ] GPU atomic min/max for mip level at each tile coordinate (deduplication on GPU)
  - [ ] Async readback: copy feedback buffer to staging, read on CPU next frame
  - [ ] CPU deduplication of readback data: unique (tile_x, tile_y, mip) requests
  - [ ] Compare requested pages against resident pages
  - [ ] Missing pages added to stream queue with priority score
  - [ ] Feedback resolution independent of render resolution (1K for any res)

### T-ENV-2.13: Virtual Texturing -- Streaming System
- **Gaps**: S12-G8
- **Effort**: HIGH (6-10 days)
- **Dependencies**: T-ENV-2.12 (feedback pass), S16 asset pipeline (async I/O)
- **Acceptance Criteria**:
  - [ ] Stream queue: priority-sorted list of pending page loads
  - [ ] Priority = visibility_weight * distance_factor + mip_weight * mip_factor + velocity_weight * prediction
  - [ ] Distance factor: 1.0 - clamp(dist/max_dist, 0, 1)
  - [ ] Mip factor: 1.0 - abs(mip_bias - 0.5) * 2.0
  - [ ] Velocity prediction: future camera position for pre-emptive loading
  - [ ] Async I/O: read page data from disk (compressed BC/ASTC/ETC2)
  - [ ] Decompress: LZ4 or similar (~0.1ms per page)
  - [ ] Max pending requests: 100 (configurable)
  - [ ] GPU upload: copy decompressed tile to physical atlas (~0.05ms per page)
  - [ ] Page table update: set physical tile coordinates + resident flag
  - [ ] LRU eviction on physical atlas full: evict oldest accessed tile
  - [ ] Fallback chain: missing page -> lower mip (always resident) -> fallback color
  - [ ] End-to-end latency: <16ms from request to resident (target with SSD)
  - [ ] Bandwidth budget: <500 unique page requests per frame typical

---

## Phase 3: Integration, Polish & Mobile Fallback (12 tasks)

### T-ENV-3.1: Weather Map System
- **Gaps**: S11-G6
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-2.2 (cloud rendering consumes weather maps)
- **Acceptance Criteria**:
  - [ ] @component WeatherStateMachine with states: Clear, PartlyCloudy, Overcast, Foggy, Storm
  - [ ] 2D weather map texture (512x512), RG channels: coverage (R), cloud_type (G)
  - [ ] Weather map generation: procedural noise-based or artist-painted
  - [ ] State transitions: blend weather maps over 30-120 seconds
  - [ ] Wind direction + speed as weather parameters
  - [ ] Cloud coverage drives cloud density remapping threshold
  - [ ] Cloud type drives cumulus vs. stratus noise parameters
  - [ ] Precipitation channel (B) reserved for future water/rain system

### T-ENV-3.2: Aerial Perspective Integration
- **Gaps**: Covered by S11-G2 (expanded)
- **Effort**: MEDIUM (2-4 days)
- **Dependencies**: T-ENV-1.1 (aerial perspective LUT), T-ENV-1.9 (terrain receives it)
- **Acceptance Criteria**:
  - [ ] Aerial perspective applied to terrain: sample LUT 3 at view_dir, distance
  - [ ] far_color = terrain_albedo * transmittance + inscatter
  - [ ] Distance thresholds match terrain LOD levels for consistent blending
  - [ ] Horizon blend: terrain farthest LOD blends into sky horizon color
  - [ ] No visible seam between terrain and sky at horizon

### T-ENV-3.3: Layered Fog System
- **Gaps**: Covered by S11-G2 (expanded)
- **Effort**: MEDIUM (2-4 days)
- **Dependencies**: T-ENV-1.5 (froxel density)
- **Acceptance Criteria**:
  - [ ] Ground fog: dense, low altitude, sharp falloff (mist, morning fog)
  - [ ] Mid haze: subtle, mid altitude, gradual falloff (distant mountains)
  - [ ] High haze: thin, high altitude (upper atmosphere blue haze)
  - [ ] Each layer: independent color, density, height, falloff
  - [ ] Layers composited additively into froxel density field
  - [ ] Per-layer quality toggle (disable ground fog on mobile)

### T-ENV-3.4: Full S4 Light Integration for Froxels
- **Gaps**: Covered by S11-G2 (expanded)
- **Effort**: HIGH (5-8 days)
- **Dependencies**: T-ENV-1.5 (froxel scattering), S4 lighting (clustered light list)
- **Acceptance Criteria**:
  - [ ] Point light scattering through froxels: light cone visible in fog
  - [ ] Spot light scattering through froxels: beam visible in fog/dust
  - [ ] Each froxel evaluates N nearest lights from S4 clustered list
  - [ ] Shadow map integration: froxel checks shadow map for each light
  - [ ] Phase function per light type (isotropic fog, forward-scattering mist)
  - [ ] Performance: bounded by max lights per froxel (S4 cluster culling)

### T-ENV-3.5: Performance Budget & LOD System
- **Gaps**: S11-G13
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: All Phase 1+2 rendering features
- **Acceptance Criteria**:
  - [ ] Quality tier definitions: Low/Medium/High/Ultra for each feature
  - [ ] GPU time targets per feature per tier (from S11 spec section 9.1)
  - [ ] Distance-based LOD: sky technique changes with distance (near/mid/far/horizon)
  - [ ] Adaptive froxel resolution: match render resolution scale
  - [ ] View-region optimization: reduce Z-slices indoors
  - [ ] Feature auto-disable: disable clouds on low-end GPUs, replace with billboard
  - [ ] Total budget: <2ms (1080p Medium) for S11 + S12 combined

### T-ENV-3.6: Mobile Fallback Quality Profile
- **Gaps**: S11-G14
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-3.5 (quality tiers)
- **Acceptance Criteria**:
  - [ ] Mobile profile: Low-quality everywhere, total <3ms GPU
  - [ ] Sky: Aerial perspective LUT only (no Bruneton sky rendering)
  - [ ] Fog: 32x24x16 froxels, no temporal reprojection
  - [ ] Clouds: 32 steps, half resolution, no reprojection, no shadows
  - [ ] Water: Gerstner only (no FFT), simplified shading
  - [ ] Terrain: 256x256 splat maps, 6 clipmap levels, 2m finest spacing
  - [ ] Foliage: billboard only, 100K max instances
  - [ ] No virtual texturing on mobile
  - [ ] Feature cap per tier enforced in quality settings

### T-ENV-3.7: World Partition Streaming
- **Gaps**: Covered by S12 (world streaming)
- **Effort**: HIGH (6-10 days)
- **Dependencies**: S15 component_store.rs, S1 frame graph
- **Acceptance Criteria**:
  - [ ] @chunk: cell definition with size, overlap params
  - [ ] @streamable: streaming priority, keep_loaded flags
  - [ ] @loading_priority: visibility_weight, player_velocity_weight
  - [ ] @unloadable: min_age, save_state policy
  - [ ] Cell state machine: unloaded -> loading -> loaded -> activated
  - [ ] Async cell loading: terrain -> height data -> GPU upload
  - [ ] Priority computation: distance + velocity prediction + LOD bonus
  - [ ] Cell activation triggers: clipmap update, foliage instance merge

### T-ENV-3.8: FFT Multi-Cascade Ocean
- **Gaps**: Covered by S12 (FFT multi-cascade)
- **Effort**: MEDIUM (4-6 days)
- **Dependencies**: T-ENV-2.8 (FFT single cascade)
- **Acceptance Criteria**:
  - [ ] Near cascade: 512x512 FFT, 100m patch, covers 0-200m
  - [ ] Mid cascade: 256x256 FFT, 500m patch, covers 200-1000m
  - [ ] Far cascade: 128x128 FFT, 2000m patch, covers 1000-5000m
  - [ ] Blended transitions between cascades (no visible seam)
  - [ ] Total FFT cost: <1.0ms for all three cascades

### T-ENV-3.9: Underwater Post-Process
- **Gaps**: Covered by S12 (underwater effects)
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-1.8 (water shading)
- **Acceptance Criteria**:
  - [ ] Caustics: screen-space approach, project sunlight through water heightfield
  - [ ] Absorption: wavelength-dependent transmittance (red/green/blue coefficients)
  - [ ] Blue-green color shift applied to scene underwater
  - [ ] Fog/distance fade based on water turbidity
  - [ ] Screen-space distortion from surface waves (refraction)
  - [ ] Performance: <0.2ms full-screen pass

### T-ENV-3.10: Shoreline Interaction
- **Gaps**: Covered by S12 (shoreline)
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-1.7 (Gerstner), T-ENV-1.9 (terrain heightfield)
- **Acceptance Criteria**:
  - [ ] @component ShorelineConfig: morph_distance (10m), wave_absorption_distance (50m)
  - [ ] Water clamping: water surface clamped to terrain height at shoreline
  - [ ] Vertex morphing: water vertices snap to terrain height over morph_distance
  - [ ] Wave shoaling: wave amplitude decreases as depth decreases
  - [ ] Terrain heightmap sampled from water vertex shader for interaction
  - [ ] No visible seam at water/land boundary (water overlaps terrain slightly)

### T-ENV-3.11: Foliage Wind Animation
- **Gaps**: Covered by S12 (foliage wind)
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-1.11 (foliage instancing), T-ENV-3.1 (weather system for wind)
- **Acceptance Criteria**:
  - [ ] Wind params uniform: direction (vec2), strength (float), frequency (float)
  - [ ] Primary sway: sinusoidal displacement scaled by height above ground
  - [ ] Turbulence: noise-based secondary displacement
  - [ ] Wind response per foliage type (grass bends more than trees)
  - [ ] Weather system integration: wind speed + direction from weather state
  - [ ] Calm/breezy/storm blend: smooth interpolation between wind states

### T-ENV-3.12: Advanced Foam Simulation (Advection-Based)
- **Gaps**: Covered by S12 (foam simulation)
- **Effort**: MEDIUM (3-5 days)
- **Dependencies**: T-ENV-2.9 (crest + shore foam foundation)
- **Acceptance Criteria**:
  - [ ] Advection-based foam: foam texture advected by wave velocity field
  - [ ] Particle foam for boat wakes, impacts (particle system integration)
  - [ ] Foam accumulation zones (lee of rocks, shoreline eddies)
  - [ ] Foam dissipation: decay_rate per foam type (crest vs. simulated)
  - [ ] enable_simulation toggle (off by default, performance cost)
