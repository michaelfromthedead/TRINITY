# Phase 6: Shadow Atlas and Filtering Architecture

## Current Status: ABSENT (0/5 tasks real)

No Rust atlas, no shadow_common.wgsl, no shadow_filter_pcf/pcss.wgsl. Python `ShadowAtlas`, `PCFFilter`, `PCSSFilter` exist as reference. PCF is inlined in `shadow_csm.wgsl` and `pbr.frag.wgsl`.

## Architecture

### Shadow Atlas (atlas.rs)

The shadow atlas packs multiple shadow maps into a single large texture. Python `ShadowAtlas` implements best-fit rectangle packing.

**Atlas parameters**:
- Default resolution: 4096x4096
- Tile sizes: 256, 512, 1024, 2048 (scaled by per-light resolution_scale)
- Max tile count: depends on size mix (e.g., 64x 512-tiles in a 4096 atlas)

**Allocation algorithm** (matches Python):
1. Maintain free-list of rectangles: `[(x, y, width, height)]`
2. On allocate: find best-fit rectangle (minimum wasted space)
3. Split remaining space into up to 2 new free rectangles
4. On deallocate: add rectangle back to free list (optional merge)
5. On overflow: gracefully degrade by reducing lowest-priority light's resolution

**Rust struct**:
```
struct ShadowAtlas {
    resolution: u32,
    free_rects: Vec<Rect>,
    slots: Vec<ShadowAtlasSlot>,
}

struct ShadowAtlasSlot {
    x: u32, y: u32,    // pixel offset in atlas
    width: u32, height: u32,
    light_id: u32,
    shadow_map_type: ShadowMapType,
}
```

### ShadowTileInfo GPU Buffer

Per-light metadata for shader sampling:

```
struct ShadowTileInfo {
    offset: vec2<f32>,     // UV offset in atlas (pixels / atlas_resolution)
    scale: vec2<f32>,      // UV scale (tile_size / atlas_resolution)
    layer_index: u32,      // array layer for cube shadows
    shadow_type: u32,      // 0=CSM, 1=cube, 2=spot
    _pad: f32,
}
```

Built from `ShadowAtlasSlot` data each frame when allocation changes.

### shadow_common.wgsl

Shared shadow types and helpers, consumed by all shadow filter modules:

```
struct ShadowTileInfo { ... }   // from above

struct BiasParams {
    constant_bias: f32,
    slope_bias: f32,
    normal_bias: f32,
    _pad: f32,
}

fn slope_scaled_bias(cos_theta: f32, slope_bias: f32) -> f32 {
    return slope_bias * tan(acos(min(cos_theta, 0.9999)));
}

fn compute_bias(normal: vec3<f32>, light_dir: vec3<f32>, params: BiasParams) -> f32 {
    let cos_theta = max(dot(normal, light_dir), 0.0);
    return params.constant_bias + slope_scaled_bias(cos_theta, params.slope_bias)
           + params.normal_bias * (1.0 - cos_theta);
}
```

### shadow_filter_pcf.wgsl

Separate PCF sampling module supporting multiple kernel sizes:

| Kernel | Radius | Samples | Use |
|--------|--------|---------|-----|
| 2x2 | 1 | 4 | Ultra-fast, low quality |
| 3x3 | 1 | 9 | Default quality |
| 5x5 | 2 | 25 | High quality |
| 7x7 | 3 | 49 | Max quality |

**Pattern options**: grid (default), Poisson disk, Vogel disk

Python `PCFFilter` supports all patterns. The pre-computed Poisson disk samples from Python `_poisson_disk()` (16 samples) should be replicated in WGSL as a constant array.

### shadow_filter_pcss.wgsl

Percentage-Closer Soft Shadows with 3 steps:

1. **Blocker search**: Average depth of blockers within search radius
2. **Penumbra estimation**: `penumbra = (receiver_z - avg_blocker_z) / avg_blocker_z * light_size`
3. **Variable PCF**: PCF with kernel radius proportional to penumbra

Python `PCSSFilter` parameters:
- blocker_search_samples: 16
- pcf_samples: 32
- light_size: 1.0 (configurable)
- max_filter_radius: 0.1 (clamp)

### Integration with CSM

The CSM pipeline from Phase 4 uses its own PCF (in `shadow_csm.wgsl`). For the atlas-based approach:
- CSM cascades can continue using array textures with inline PCF
- Point/spot shadows use atlas + shared filter modules
- `shadow_common.wgsl` types are used by both paths for consistency

### Test Plan

- Atlas allocation: 100-frame stress test, measure fragmentation
- PCF: compare kernel output against Python PCFFilter at 10+ test points
- PCSS: verify wider penumbras with increasing occluder distance
- Hardware PCF: confirm `textureSampleCompare` with `sampler_comparison`
