# PHASE_2_ARCH.md -- Clouds, FFT Ocean & Virtual Texturing

## Overview

Phase 2 adds three major subsystems that are entirely missing from the codebase: volumetric cloud rendering, FFT ocean simulation, and virtual texturing. Unlike Phase 1 tasks which could leverage existing world-layer data, Phase 2 requires building both world-layer data structures (where needed) and complete rendering pipelines.

**Total effort:** ~60-90 person-days
**Files to create:** ~20 new Python files + ~10 new WGSL shaders

---

## T-ENV-2.1: Cloud Noise Textures

### World Layer (New)

Create `engine/world/environment/cloud_noise.py` (or bake offline):

```python
class CloudNoiseGenerator:
    """Generates cloud noise textures for GPU consumption."""
    @staticmethod
    def worley_noise_3d(resolution: int, octaves: int) -> np.ndarray:
        """3D Worley noise (cell distance) at 32^3, RG8 format, 2 octaves."""
    @staticmethod
    def perlin_worley_fbm(resolution: int, octaves: int) -> np.ndarray:
        """Perlin-Worley FBM (4-5 octaves) for base shape."""
    @staticmethod
    def detail_noise(resolution: int, octaves: int) -> np.ndarray:
        """Detail Worley FBM (2-3 octaves) at 16^3, R8."""
```

**Two approaches:**
- **Option A (CPU at init):** Generate at engine startup using Python numpy. Portable, adds ~100ms to init time. Total memory <2MB.
- **Option B (baked):** Pre-generate via asset pipeline, store as .trinity_noise format. Production path.

**Key properties:**
- 3D Worley noise at 32^3 (RG8, 2 octaves) -- provides cloud shape base
- Detail noise at 16^3 (R8) -- provides wispy edge detail
- Perlin-Worley FBM (4-5 octaves) -- base shape coverage
- Tiling: 4-8km tile size with cross-fade at boundaries
- Texel animation: detail noise scrolls at different rate than base

---

## T-ENV-2.2: Cloud Ray Marching Pass

### World Layer Inputs

- `WeatherParameters.cloud_density` from `weather.py`
- `WeatherParameters.wind_direction`, `wind_speed`
- Sun direction from TOD
- Noise textures from T-ENV-2.1

### Implementation

Create `engine/rendering/atmosphere/clouds.py`:

```python
class CloudLayerConfig:
    min_height: float = 1000.0  # 1-8km
    thickness: float = 4000.0   # 2-8km
    coverage: float = 0.5       # 0-1, from weather map in Phase 3
    extinction: float = 0.5
    powder_factor: float = 2.0
    albedo: float = 0.9

class CloudMarchPass:
    """Full-screen compute pass for cloud ray marching."""
    quality_configs = {
        "Low": {"steps": 32, "half_res": True},
        "Medium": {"steps": 64},
        "High": {"steps": 128},
        "Ultra": {"steps": 256},
    }
```

Create `crates/renderer-backend/shaders/clouds/cloud_march.wgsl`:

```wgsl
// Full-screen compute: one thread per pixel
// For each pixel:
//   1. Compute view ray direction from camera
//   2. March through cloud slab (min_height to min_height + thickness)
//   3. At each step:
//      a. Sample base shape noise + detail noise at world-space position
//      b. Density remapping: shift raw noise by coverage, erode by detail
//      c. Beer's law: segment_transmittance = exp(-density * step_size * extinction)
//      d. Powder effect: 1.0 - exp(-density * powder_factor)
//      e. Multi-scattering approx: single / max(1 - albedo*(1-T), 0.001)
//      f. Accumulate radiance + transmittance
//   4. Early termination when transmittance < 0.01
//   5. Depth-aware: stop behind solid geometry
// Output: cloud radiance + transmittance
```

**Cumulus vs. stratus:** Controllable via noise parameter modulation. Cumulus uses sharper density remapping (higher noise contrast). Stratus uses smoother density (lower noise contrast). Cloud type selector from weather map (Phase 3).

---

## T-ENV-2.3: Cloud Lighting (Beer + Powder + Multi-Scatter)

Part of the cloud ray marching shader (T-ENV-2.2), specified here for clarity.

**Beer's law:** `transmittance = exp(-density * distance)`
- Pure exponential extinction through cloud density
- Verified against reference implementation

**Powder effect:** `brightening = 1.0 - exp(-density * powder_factor)`
- Brightens cloud edges where density approaches zero
- Creates the characteristic "lit from within" look at cloud boundaries

**Multi-scattering approximation:**
- `forward_scatter = single_scatter / max(1 - albedo * (1 - transmittance), 0.001)`
- Models light bouncing within the cloud volume
- Brighter cloud interiors than single-scattering alone
- Energy conservation: output radiance <= input radiance

**Configurable parameters:**
- Extinction coefficient (density multiplier)
- Powder factor (edge brightness)
- Albedo (cloud whiteness, ~0.9 typical)

---

## T-ENV-2.4: Cloud Shadows on Terrain

### Implementation

After the cloud density field is computed, project shadows onto terrain:

```wgsl
// For each terrain shading point:
//   1. Ray march from terrain point toward sun direction
//   2. Accumulate density through cloud layer (2-8km only)
//   3. Shadow factor = exp(-density_sum * shadow_extinction)
//   4. Apply to terrain: sun_irradiance *= cloud_shadow_factor
```

**Temporal accumulation:** `blend 0.9 * previous + 0.1 * current` for stability.
**Wind drift:** Shadow moves with cloud drift (wind direction * speed).
**Performance target:** <0.3ms at 1080p via 16-step ray march per shading point.

---

## T-ENV-2.5: God Rays (Volumetric Light Shafts)

### Implementation

Two-component approach:

1. **Volumetric base:** Reuse froxel scattering from T-ENV-1.5. Add sun transmittance per froxel (ray march toward sun, accumulate density from clouds/fog).

2. **Screen-space detail:** Radial blur from sun screen position on a visibility buffer.
   - Visibility buffer: 0=shadowed, 1=lit (from shadow maps or cloud shadow)
   - Radial samples: 32/64 for Medium/High
   - Radial blur extends light rays across screen

3. **Hybrid blend:** `lerp(volumetric, screen_space, detail_blend_factor)`

4. **Compositing:** `final_color += godray_color * intensity` (additive)

Create `crates/renderer-backend/shaders/atmosphere/god_ray.wgsl`.

---

## T-ENV-2.6: Temporal Reprojection for Fog and Clouds

### Implementation

Create `engine/rendering/atmosphere/temporal_reprojection.py`:

```python
class TemporalReprojectionConfig:
    blend_factor: float = 0.1  # new frame weight
    depth_rejection_threshold: float = 0.01
    radiance_rejection_threshold: float = 0.5

class FroxelReprojection:
    """Reprojects previous frame froxels using current camera matrices."""
    def reproject(self, prev_froxels: Texture, prev_view: mat4, 
                  curr_view: mat4, depth: Texture) -> Texture: ...

class CloudReprojection:
    """Reprojects previous frame cloud buffer."""
    def reproject(self, prev_cloud: Texture, ...) -> Texture: ...
```

**Pass structure:**
1. Velocity buffer: screen-space motion vectors from camera movement
2. Froxel reprojection: reproject using previous camera matrices
3. Cloud reprojection: same pattern for cloud buffer
4. Blend: `result = lerp(previous_reprojected, current, 0.1)`
5. Rejection: skip samples with large depth discontinuity or large radiance change
6. Full reset on camera cut or teleport (detected via large camera position delta)
7. Upscale: bilateral filter from half-res to full-res

**Quality saving:** 2-4x effective resolution at similar cost. Temporal stability goal: static camera = stable image within 10 frames.

---

## T-ENV-2.7: LUT Cooking Pipeline (S16 Integration)

### World Layer (New)

Create `engine/rendering/atmosphere/lut_pipeline.py`:

```python
class CookedLUT:
    """Serializable LUT with header + RGBA16F data."""
    format_magic = b"TRINITY_LUT"
    version: int = 1
    def save(self, path: str, data: np.ndarray, metadata: dict): ...
    def load(self, path: str) -> tuple[np.ndarray, dict]: ...
```

**File format:** `.trinity_lut` -- header (magic bytes + version + dimensions + format) followed by raw RGBA16F data.

**LUT versioning:** When sun model parameters change (Rayleigh coefficient, ozone, etc.), LUT version increments and CPU recomputation occurs.

**Fallback:** If cooked LUT not found on disk, CPU recompute at init (same as T-ENV-1.1 path).

---

## T-ENV-2.8: FFT Ocean Compute Shader

### World Layer (New)

Create `engine/world/environment/ocean.py`:

```python
class FFTOceanConfig:
    fft_size: int = 256          # power of 2
    patch_size: float = 500.0    # meters
    wind_speed: float = 10.0     # m/s
    wind_direction: vec2 = (1.0, 0.0)
    chop_amount: float = 1.0
    phillips_constant: float = 0.0001  # scales wave height

class PhillipsSpectrum:
    """Generates h0(K) frequency domain initial wave spectrum."""
    def generate(self, config: FFTOceanConfig, seed: int) -> np.ndarray:
        """Complex-valued 2D array of size fft_size x fft_size."""
        # For each K (wave vector):
        #   L = wind_speed^2 / g
        #   phillips = A * exp(-1/(kL)^2) / k^4 * |dot(K, wind_dir)|^2
        #   h0 = 1/sqrt(2) * (gaussian_random + i * gaussian_random) * sqrt(phillips)
```

### Implementation

Create `engine/rendering/water/fft_ocean.py`:

```python
class FFTOceanPass:
    """GPU FFT ocean simulation."""
    def build(self, fg: FrameGraph) -> list[PassNode]:
        # Pass 1: Time evolution (compute)
        # Pass 2: IFFT rows (compute)
        # Pass 3: IFFT columns (compute)
        # Output: heightfield texture (R32F)
```

Create `crates/renderer-backend/shaders/water/fft_ocean.wgsl`:

```wgsl
// Pass 1 - Time Evolution:
// h(K,t) = h0(K) * exp(i*w*t) + conj(h0(-K)) * exp(-i*w*t)
// where w = sqrt(g * k)

// Pass 2 - IFFT Rows:
// 1D IFFT along X for each row using Stockham radix-2

// Pass 3 - IFFT Columns:
// 1D IFFT along Z for each column using Stockham radix-2

// Output: heightfield (R32F)
// Optional: slope field for normal computation
// Optional: displacement field for choppy waves
```

**Verification:** `IFFT(FFT(x)) == x` within floating point precision.
**Performance:** <0.3ms for 256x256 FFT.

---

## T-ENV-2.9: Foam Generation (Crest + Shore)

### Implementation

Create `engine/rendering/water/foam.py` and `crates/renderer-backend/shaders/water/foam.wgsl`:

**Crest foam:** From wave Jacobian. When `J < 0`, wave has folded (broken). `foam = clamp(1 - J/threshold, 0, 1)`. Decays over time: `foam(t) = max(0, foam - decay_rate * dt)`.

**Shore foam:** Based on shoreline distance. Band width configurable (~5m typical). Combined with wave height for breaking wave foam at shore.

**Foam mask:** Full-resolution R8 render target, composited as additive overlay with foam noise texture.

**Foam texture:** Procedural noise pattern for detail, sampled at surface UVs.

**Simulated foam:** Placeholder for Phase 3 advection-based system. For Phase 2, crest + shore detection is sufficient.

---

## T-ENV-2.10: Virtual Texturing -- Page Table

### World Layer (New)

Create `engine/rendering/texturing/virtual_texturing.py`:

```python
class VirtualTextureConfig:
    virtual_size: int = 131072      # 128K texels
    tile_size: int = 128            # texels per tile
    page_table_size: int = 1024     # 1024x1024 = ~1M entries

class PageTableEntry:
    """RGBA16Uint per entry."""
    physical_x: int      # u16: tile position in physical atlas
    physical_y: int      # u16: tile position in physical atlas
    mip_level: int       # u8: which mip level this tile belongs to
    flags: int           # u8: bit 0=resident, bit 1=requested, bit 2=streaming, bit 3=invalid

class PageTable:
    """1024x1024 RGBA16Uint texture (8MB)."""
    def sample(self, vt_uv: vec2, layer: int) -> vec4:
        """Shader sampling function. Returns fallback_color for non-resident pages."""
    def update(self, entries: dict[tuple[int,int,int], PageTableEntry]): ...
```

**Virtual address space:** 128K x 128K = 1024 x 1024 virtual tiles (128px each). Page table entry = 8 bytes (4 x u16), total page table = 8MB.

---

## T-ENV-2.11: Virtual Texturing -- Physical Atlas

### Implementation

```python
class PhysicalAtlas:
    """16K x 16K texture array, divided into tile slots."""
    atlas_size: int = 16384
    tile_size: int = 128
    border_size: int = 2  # texel padding per side
    num_tiles: int = (16384 // 128) ** 2  # 128^2 = 16,384 slots
    mip_levels: int = 13  # log2(16384/4)
```

**Physical layout:**
- 16K x 16K atlas = 256M texels
- Each tile slot: (128 + 4) x (128 + 4) = 132 x 132 with 2-texel border padding
- Mip chain stored as separate page tables for each mip level
- LRU eviction queue: free list + deque for O(1) tile slot management
- Upload: staging buffer -> atlas copy
- Evicted tiles flushed if dirty

---

## T-ENV-2.12: Virtual Texturing -- Feedback Pass

### Implementation

```python
class FeedbackPass:
    """Determines which virtual tiles are visible this frame."""
    feedback_buffer: Texture  # 1024x1024 R32Uint
    def execute(self, encoder: CommandEncoder, virtual_uv_texture: Texture): ...
    def readback(self) -> set[tuple[int, int, int]]:
        """Async readback, CPU deduplication."""
```

**Algorithm:**
1. Render virtual UV coordinates to feedback buffer
2. For each visible pixel: encode (tile_x, tile_y, mip) using GPU atomics
3. GPU min/max deduplication: atomic min/max for mip at each tile coordinate
4. Async readback: copy feedback -> staging -> CPU next frame
5. CPU deduplication: unique (tile_x, tile_y, mip) requests
6. Compare against resident pages, enqueue missing pages

---

## T-ENV-2.13: Virtual Texturing -- Streaming System

### Implementation

```python
class VTStreamQueue:
    """Priority-sorted queue of pending page loads."""
    max_pending: int = 100
    
    def compute_priority(self, request: PageRequest) -> float:
        """Priority = visibility_weight * distance_factor 
                     + mip_weight * mip_factor 
                     + velocity_weight * prediction"""
    
    def update(self, requested_pages: set[PageRequest], 
               camera_pos: vec3, camera_velocity: vec3): ...
    
    def process(self):
        """Async I/O: read compressed page -> decompress (LZ4) -> upload to atlas -> update page table."""
```

**Priority factors:**
- Distance factor: `1.0 - clamp(dist/max_dist, 0, 1)` -- nearer tiles load first
- Mip factor: `1.0 - abs(mip_bias - 0.5) * 2.0` -- desired mip level weighted higher
- Velocity prediction: pre-emptively load tiles in direction of camera movement

**Pipeline:**
1. Read compressed page from disk (BC/ASTC/ETC2)
2. LZ4 decompress (~0.1ms per page)
3. Upload to physical atlas (~0.05ms per page)
4. Update page table entry: set physical coordinates + resident flag
5. LRU eviction when atlas full: evict oldest accessed tile
6. Fallback chain: missing page -> lower mip (always resident) -> fallback color

**Target latency:** <16ms from request to resident (with SSD). Bandwidth budget: <500 unique page requests per frame.

---

## Phase 2 Performance Budget

| Feature | GPU Time Budget | Priority |
|---------|----------------|----------|
| Cloud Ray Marching (64 steps) | <0.5ms @ 1080p | High |
| Cloud Shadows | <0.3ms | Medium |
| God Rays | <0.2ms | Medium |
| Temporal Reprojection (fog + clouds) | <0.3ms | High |
| FFT Ocean (256x256) | <0.3ms | High |
| Foam Generation | <0.1ms | Low |
| Virtual Texturing Feedback | <0.1ms | High |
| Virtual Texturing Upload | ~0.05ms per page | Medium |
| Total Phase 2 features | <2.0ms | |

---

## Key Technical Decisions for Phase 2

1. **Cloud noise generated at init (Option A)** for simplicity. Baked option (Option B) is a future optimization.

2. **Single-scattering + multi-scattering approximation** for cloud lighting. Full multi-scattering would require multiple ray marches, which is too expensive for real-time.

3. **FFT over spectrum synthesis on GPU** for ocean waves. The Phillips spectrum + IFFT pipeline is the standard approach for open water. Gerstner (Phase 1) handles near-shore detail.

4. **Virtual texturing over mega-textures.** Virtual texturing provides full-resolution terrain texturing without streaming 20GB of unique texture data. Page table indirection is well-understood and supported by modern GPU hardware.

5. **Temporal reprojection before costing optimization.** Phase 2 adds the reprojection infrastructure. Phase 3 optimizes and tunes it.

6. **LUT cooking is a simple serialization format.** No complex asset pipeline integration needed for Phase 2 -- just header + raw data with version check.
