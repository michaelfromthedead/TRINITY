# Investigation: engine/rendering/lighting

**Lines**: 4,470 (core files analyzed)
**Classification**: REAL

## Summary
The lighting subsystem is a COMPREHENSIVE CPU-SIDE IMPLEMENTATION with real algorithms for shadow mapping (CSM, cube, spot), shadow filtering (PCF, PCSS, VSM, ESM, contact shadows), clustered light culling (3D froxels), and global illumination (SH light probes, DDGI, reflection probes). All math and data structures are fully implemented, but GPU execution is stubbed - there are no actual GPU texture handles, shader dispatch, or render passes.

## Files Analyzed (Task Specification)
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `gi_ddgi.py` | 843 | Real | Full DDGI with octahedral encoding, Chebyshev visibility |
| `shadow_filtering.py` | 796 | Real | PCF, PCSS, VSM, ESM, contact shadows with math |
| `shadows.py` | 784 | Real | CSM with cascade stabilization, cube/spot shadow maps, atlas |
| `gi_probes.py` | 779 | Real | SH L2 (27 coefficients), probe grids, lightmaps, reflection probes |
| `light_types.py` | 650 | Real | 7 light types with full attenuation math |
| `light_culling.py` | 618 | Real | Full froxel grid + sphere/AABB culling |

**Total: 4,470 lines** analyzed

## Lighting Components

### Light Types (light_types.py)
- `DirectionalLight` - CSM cascade config, angular diameter for PCSS
- `PointLight` - Position, radius, smooth inverse-square falloff
- `SpotLight` - Inner/outer angle, angular attenuation with smoothstep
- `RectAreaLight` - LTC-referenced, width/height, two-sided
- `DiskAreaLight` - LTC-referenced, disk radius
- `IESLight` - Full IES profile parsing with bilinear sampling
- `SkyLight` - Cubemap path, rotation, mip levels for roughness

### Shadow Mapping (shadows.py)
- `CascadedShadowMap` - 1-4 cascades, logarithmic split scheme, stabilization
- `CubeShadowMap` - 6 faces with proper direction/up vectors
- `SpotShadowMap` - Single frustum matching cone angle
- `ShadowAtlas` - Best-fit rectangle packing, defragmentation

### Shadow Filtering (shadow_filtering.py)
- `PCFFilter` - Grid, Poisson disk, Vogel disk patterns
- `PCSSFilter` - Blocker search + variable penumbra
- `VSMFilter` - Chebyshev inequality with light bleeding reduction
- `ESMFilter` - Exponential depth with configurable exponent
- `ContactShadowFilter` - Screen-space ray march interface

### Clustered Lighting (light_culling.py)
- `FroxelGrid` - 3D grid with exponential depth slicing
- `ClusteredLightCuller` - Sphere/AABB intersection, per-froxel light lists
- `LightList` - GPU-ready offset/count structure

### Global Illumination (gi_probes.py + gi_ddgi.py)
- `SphericalHarmonics` - L2 SH with 27 coefficients, evaluate/add_sample
- `LightProbe` - SH-based, influence radius, baking with Fibonacci spiral
- `ProbeGrid` - 3D grid with trilinear interpolation
- `IrradianceVolume` - Edge blending wrapper
- `BakedLightmap` - Per-texel irradiance with bilinear sampling
- `ReflectionProbe` - Parallax box correction, blend factor
- `DDGIProbe` - Octahedral irradiance + visibility encoding
- `DDGIProbeGrid` - Scrolling grid, prioritized updates
- `DDGIUpdatePass` - Ray tracing interface with Fibonacci spiral
- `DDGILookup` - Trilinear + Chebyshev visibility weighting

## Implementation

### Real shadow maps? YES (CPU math only)
- CSM: Full cascade split computation (logarithmic), frustum corner extraction, view matrix from light direction, cascade stabilization (texel snapping)
- Cube: 6 face matrices with proper orientations
- Spot: Projection matching cone angle
- Atlas: Rectangle packing algorithm with defragmentation

### Real cascade shadow maps? YES (CPU math only)
```python
# shadows.py:218-245
def _compute_cascade_splits(self, near: float, far: float) -> list[float]:
    # Use provided cascade distances if available
    if self._cascade_distances and len(self._cascade_distances) >= self.cascade_count:
        return self._cascade_distances[:self.cascade_count]
    # Compute using logarithmic distribution
    lambda_param = 0.75  # Blend factor between linear and logarithmic
    for i in range(self.cascade_count):
        t = (i + 1) / self.cascade_count
        log_split = near * math.pow(far / near, t)
        linear_split = near + (far - near) * t
        split = lambda_param * log_split + (1 - lambda_param) * linear_split
```

### Real DDGI? YES (CPU simulation only)
```python
# gi_ddgi.py:131-153 - Octahedral encoding
def _direction_to_octahedral(self, direction: Vec3) -> Vec2:
    d = direction.normalized()
    inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z))
    ox = d.x * inv_l1
    oy = d.y * inv_l1
    if d.z < 0:  # Wrap negative hemisphere
        ox = (1.0 - abs(oy)) * (1.0 if ox >= 0 else -1.0)
        oy = (1.0 - abs(ox)) * (1.0 if oy >= 0 else -1.0)
    return Vec2(ox * 0.5 + 0.5, oy * 0.5 + 0.5)
```

### GPU execution? NO - STUBBED
```python
# shadows.py:72-74
# GPU resource handles (would be actual GPU resources in production)
_texture_handle: int = 0
_depth_handle: int = 0

# gi_probes.py:669-670
# Placeholder: return a default color
# Real implementation would sample the cubemap texture
return Vec3(0.5, 0.5, 0.5)

# shadow_filtering.py:703-704
# In production, this would:
# 1. Ray march from shading point toward light in screen space
# Placeholder return
return ShadowSample(visibility=1.0)
```

## Verdict
**PARTIAL IMPLEMENTATION** - Sophisticated CPU-side algorithms with complete math

The lighting system has production-quality mathematics and data structures for:
- All industry-standard shadow techniques
- Clustered deferred/forward lighting infrastructure
- Both baked and dynamic GI systems

However, NO GPU CODE EXISTS:
- No shader generation
- No render pass submission
- No texture/buffer creation
- All "GPU resource handles" are placeholder integers
- DDGI ray tracing takes a callback `trace_func` but no implementation exists
- Reflection probe sampling returns hardcoded `Vec3(0.5, 0.5, 0.5)`

## Evidence

### Shadow Filtering Math (Real)
```python
# shadow_filtering.py:428-446 - PCSS penumbra estimation
def _estimate_penumbra(self, receiver_depth: float, blocker_depth: float) -> float:
    if blocker_depth <= 0 or blocker_depth >= receiver_depth:
        return 0.0
    # Penumbra width = (d_receiver - d_blocker) * light_size / d_blocker
    return (receiver_depth - blocker_depth) / blocker_depth
```

### Spherical Harmonics (Real)
```python
# gi_probes.py:69-95 - L2 SH evaluation with proper basis functions
def evaluate(self, direction: Vec3) -> Vec3:
    y0 = 0.282095  # 1/(2*sqrt(pi))
    y1 = 0.488603 * d.y    # sqrt(3/(4*pi)) * y
    y6 = 0.315392 * (3 * d.z * d.z - 1)  # sqrt(5/(16*pi)) * (3z^2 - 1)
```

### GPU Stubs (Not Real)
```python
# gi_probes.py:665-670
def sample(self, world_pos: Vec3, reflection_dir: Vec3, roughness: float = 0.0) -> Vec3:
    corrected_dir = self._parallax_correct(world_pos, reflection_dir)
    # In production: sample cubemap at corrected direction
    # Placeholder: return a default color
    return Vec3(0.5, 0.5, 0.5)
```

## Assessment

| Component | Math Complete | Data Structures | GPU Integration |
|-----------|--------------|-----------------|-----------------|
| Light Types | YES | YES | N/A (CPU-only) |
| Shadow Maps | YES | YES | NO (stub handles) |
| Shadow Filters | YES | YES | NO (CPU sampling only) |
| Froxel Culling | YES | YES | NO (no upload) |
| SH Probes | YES | YES | NO (no cubemap) |
| DDGI | YES | YES | NO (callback stub) |
| Lightmaps | YES | YES | NO (no textures) |

This is a **high-quality lighting reference implementation** that could drive a real renderer, but requires all GPU integration work to be built on top.
