# SUMMARY: engine/rendering/lighting

## Metrics Table

| Metric | Value |
|--------|-------|
| Total Lines | 4,470 |
| Files Analyzed | 6 |
| Classification | REAL (CPU) / STUBBED (GPU) |
| Math Complete | 100% |
| Data Structures | 100% |
| GPU Integration | 0% |
| Light Types | 7 |
| Shadow Techniques | 4 (CSM, Cube, Spot, Atlas) |
| Filter Techniques | 5 (PCF, PCSS, VSM, ESM, Contact) |
| GI Techniques | 4 (SH Probes, DDGI, Lightmaps, Reflection Probes) |

## Algorithm Inventory

| Algorithm | File | Lines | Status | Notes |
|-----------|------|-------|--------|-------|
| Cascaded Shadow Maps | shadows.py | 218-245 | REAL | Logarithmic splits, stabilization |
| Cube Shadow Maps | shadows.py | - | REAL | 6-face matrices |
| Spot Shadow Maps | shadows.py | - | REAL | Cone-matched frustum |
| Shadow Atlas | shadows.py | - | REAL | Rectangle packing, defrag |
| PCF Filter | shadow_filtering.py | - | REAL | Grid, Poisson, Vogel |
| PCSS Filter | shadow_filtering.py | 428-446 | REAL | Blocker search, penumbra |
| VSM Filter | shadow_filtering.py | - | REAL | Chebyshev inequality |
| ESM Filter | shadow_filtering.py | - | REAL | Exponential depth |
| Contact Shadows | shadow_filtering.py | 703-704 | STUB | Ray march interface only |
| Froxel Grid | light_culling.py | - | REAL | Exponential depth slicing |
| Light Culling | light_culling.py | - | REAL | Sphere/AABB intersection |
| Spherical Harmonics L2 | gi_probes.py | 69-95 | REAL | 27 coefficients, proper basis |
| Light Probes | gi_probes.py | - | REAL | SH-based, Fibonacci baking |
| Probe Grid | gi_probes.py | - | REAL | Trilinear interpolation |
| Lightmaps | gi_probes.py | - | REAL | Per-texel irradiance |
| Reflection Probes | gi_probes.py | 665-670 | STUB | Parallax correct, no sample |
| DDGI Octahedral | gi_ddgi.py | 131-153 | REAL | Direction encoding |
| DDGI Visibility | gi_ddgi.py | - | REAL | Chebyshev weighting |
| DDGI Update | gi_ddgi.py | - | STUB | trace_func callback required |
| IES Profiles | light_types.py | - | REAL | Bilinear sampling |

## Evidence Snippets

### Cascaded Shadow Map Splits (REAL)
```python
# shadows.py:218-245
def _compute_cascade_splits(self, near: float, far: float) -> list[float]:
    lambda_param = 0.75  # Blend factor between linear and logarithmic
    for i in range(self.cascade_count):
        t = (i + 1) / self.cascade_count
        log_split = near * math.pow(far / near, t)
        linear_split = near + (far - near) * t
        split = lambda_param * log_split + (1 - lambda_param) * linear_split
```

### PCSS Penumbra Estimation (REAL)
```python
# shadow_filtering.py:428-446
def _estimate_penumbra(self, receiver_depth: float, blocker_depth: float) -> float:
    if blocker_depth <= 0 or blocker_depth >= receiver_depth:
        return 0.0
    # Penumbra width = (d_receiver - d_blocker) * light_size / d_blocker
    return (receiver_depth - blocker_depth) / blocker_depth
```

### Spherical Harmonics L2 Basis (REAL)
```python
# gi_probes.py:69-95
def evaluate(self, direction: Vec3) -> Vec3:
    y0 = 0.282095  # 1/(2*sqrt(pi))
    y1 = 0.488603 * d.y    # sqrt(3/(4*pi)) * y
    y6 = 0.315392 * (3 * d.z * d.z - 1)  # sqrt(5/(16*pi)) * (3z^2 - 1)
```

### DDGI Octahedral Encoding (REAL)
```python
# gi_ddgi.py:131-153
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

### GPU Texture Stubs (NOT REAL)
```python
# shadows.py:72-74
_texture_handle: int = 0
_depth_handle: int = 0
```

### Reflection Probe Placeholder (NOT REAL)
```python
# gi_probes.py:665-670
def sample(self, world_pos: Vec3, reflection_dir: Vec3, roughness: float = 0.0) -> Vec3:
    corrected_dir = self._parallax_correct(world_pos, reflection_dir)
    # Placeholder: return a default color
    return Vec3(0.5, 0.5, 0.5)
```

### Contact Shadow Stub (NOT REAL)
```python
# shadow_filtering.py:703-704
# Placeholder return
return ShadowSample(visibility=1.0)
```

## File Summary

| File | Lines | Primary Content |
|------|-------|-----------------|
| gi_ddgi.py | 843 | DDGI probes, octahedral encoding, Chebyshev visibility |
| shadow_filtering.py | 796 | PCF, PCSS, VSM, ESM, contact shadow filters |
| shadows.py | 784 | CSM, cube, spot shadow maps, atlas packing |
| gi_probes.py | 779 | SH L2, light probes, lightmaps, reflection probes |
| light_types.py | 650 | 7 light types with attenuation math |
| light_culling.py | 618 | Froxel grid, clustered culling, light lists |
