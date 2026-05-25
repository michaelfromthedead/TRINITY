# Phase 4: Ambient Occlusion -- Architecture

**Phase ID**: PHASE_4 | **Tasks**: 9 (2 [x], 4 [~], 3 [-])

---

## Module Structure

### Python: `ambient_occlusion.py` (673 lines)

```
AOSettings (dataclass)
  - method: AOMethod enum (SSAO, HBAO, GTAO)
  - radius, sample_count, power, bias, intensity
  - multi_bounce, bent_normals, temporal_filter
  - quality_levels: ray_step counts per method

AOEffect (PostProcessEffect)
  - __init__(settings)
  - execute() -> stub (empty, calls calculate() methods that return None)

--- Common ---

BilateralFilter
  - separable_blur(input_data, depth, normal, settings) -> blurred output
  - depth_weight = exp(-depth_diff^2 * sharpness)
  - normal_weight = max(0, dot(normal_a, normal_b))
  - spatial_gaussian = exp(-dx^2 / (2*sigma^2)) * exp(-dy^2 / (2*sigma^2))
  - Combined: sum(weight * input) / sum(weight)
  - CPU working: correct bilateral filtering logic

--- SSAO ---

SSAO
  - generate_hemisphere_kernel(sample_count) -> 64 random sample points
    - Seed: deterministic seed=42
    - Distribution: cosine-weighted hemisphere (z biased toward +Z)
    - Normalized to unit sphere, scaled by step * (i/64) for linear distribution
  - generate_noise() -> 4x4 random rotation vectors (16x3 matrix)
  - calculate(depth, normal, settings) -> occlusion factor (returns None)
    - Architecture: for each sample, project to UV, compare depth, range check
    - Range check: abs(depth_sample - depth_center) < radius
  - CPU reference: kernel + noise generation work, calculate returns None

--- HBAO ---

HBAO
  - generate_directions(num_directions) -> evenly spaced directions
  - calculate(depth, normal, settings) -> occlusion factor (returns None)
    - Architecture: for each direction, march N steps, find max horizon angle
    - Cosine-weighted integral over horizon angles
  - CPU reference: direction generation works, horizon marching stub

--- GTAO ---

GTAO
  - _integrate_slice(direction, depth, normal, settings) -> stub (returns 0)
  - calculate(depth, normal, settings) -> occlusion factor (returns None)
    - Architecture: dual horizon search (upper + lower angles)
    - Multi-bounce: (1+albedo)*single / (1+albedo*(1-single))
  - CPU reference: structure only, returns None
```

---

## Data Flow

```
Depth Buffer + Normal Buffer
  |
  v
[AO Method Selection] -- SSAO / HBAO / GTAO per quality preset
  |
  v
[Sample Generation] -- hemisphere kernel (SSAO) or directions (HBAO/GTAO)
  |  SSAO: 8-64 random samples, 4x4 noise tile
  |  HBAO: 2-8 rays, 3-8 steps per ray
  |  GTAO: 3-8 directions, 4-12 steps, dual horizon
  |
  v
[Occlusion Calculation] -- depth comparison + angle integration (ALL STUBS)
  |  SSAO: compare screen-space depth at sample positions (STUB)
  |  HBAO: horizon angle ray march (STUB)
  |  GTAO: dual horizon + bent normal (STUB)
  |
  v
[Bilateral Blur] -- edge-preserving spatial denoise (CPU WORKING)
  |
  v
[Bent Normals] -- directional occlusion vector (STUB)
  |
  v
[Temporal Filter] -- frame accumulation (MISSING)
  |
  v
[Composite] -- AO * scene_color blended at pipeline position 6
```

**Key observation**: AO processing has the same pattern as other effects -- complete architectural structure, kernel/sample generation works (CPU), but calculate() methods return None. BilateralFilter is the most complete component -- it actually performs convolution on numpy arrays.

---

## What Exists ([x] = 2)

| Task | Component | State |
|------|-----------|-------|
| T-PP-4.1 | SSAO compute | Hemisphere kernel (64 samples, deterministic seed), 4x4 noise, depth comparison logic. CPU reference works for kernel generation. |
| T-PP-4.1a | Bilateral blur | Complete: separable depth+normal aware blur. CPU working for reference. |

## What Is Partial ([~] = 4)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-4.1b | SSAO quality levels | Sample counts defined (8/16/32/64 per preset). No R16_FLOAT output. No performance budget. |
| T-PP-4.2 | HBAO horizon ray march | Direction generation works. calculate() returns None. No actual ray march. |
| T-PP-4.2a | HBAO falloff + quality | Falloff radius in settings. Rays x steps per quality defined. No pixel output. |
| T-PP-4.3 | GTAO bent normals | GTAO class with _integrate_slice stub. Multi-bounce formula in settings. |

## What Is Missing ([-] = 3)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-4.3a | Bent normals computation | AOSettings.bent_normals flag exists, no code path produces bent normals |
| T-PP-4.3b | Temporal filter stability | No accumulation buffer, no history frame |
| T-PP-4.4 | Phase 4 integration tests | None |

---

## SSAO Hemisphere Kernel Detail

```python
def generate_hemisphere_kernel(sample_count=64):
    rng = RandomState(seed=42)  # deterministic
    kernel = []
    for i in range(sample_count):
        # Random point in unit hemisphere
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        z = rng.uniform(0, 1)  # bias toward +Z (hemisphere)
        
        # Normalize
        length = sqrt(x*x + y*y + z*z)
        x, y, z = x/length, y/length, z/length
        
        # Scale by distance for linear distribution
        scale = i / sample_count
        scale = lerp(0.1, 1.0, scale * scale)
        kernel.append((x*scale, y*scale, z*scale))
    return kernel  # 64 samples
```

## Bilateral Filter Detail

```python
def separable_blur(input, depth, normal, sigma_s=2.0, sigma_r=0.1):
    # Horizontal pass
    for x in range(width):
        for y in range(height):
            total_weight = 0
            total_value = 0
            for dx in range(-radius, radius+1):
                sx, sy = x+dx, y
                spatial_w = exp(-dx*dx / (2*sigma_s*sigma_s))
                depth_w = exp(-(depth[sx,sy]-depth[x,y])**2 * sharpness)
                normal_w = max(0, dot(normal[sx,sy], normal[x,y]))
                weight = spatial_w * depth_w * normal_w
                total_weight += weight
                total_value += input[sx,sy] * weight
            output[x,y] = total_value / total_weight
    # Vertical pass (same pattern)
    return output
```

---

## AO Pipeline Position

The AO effect is applied at position 6 in the effect pipeline (after motion blur, before tonemapping). This matches HDR-rendering convention where AO is computed in linear HDR space and composited as a multiplicative factor on the scene color: `occluded_color = scene_color * (1 - occlusion)`.

## Interfaces

```python
# All calculate() methods return None currently
# Planned output: R16_FLOAT occlusion factor texture
class AOEffect:
    def execute(self, context, engine_data):
        # context: { depth_buffer, normal_buffer, camera }
        # engine_data: { settings, quality_preset }
        occlusion = self._calculate(depth, normal, settings)
        # occlusion: R16_FLOAT, values in [0, 1]
        # 0 = fully occluded, 1 = fully unoccluded
        occlusion = self._bilateral_blur(occlusion, depth, normal)
        # Output: engine_data["ao_output"] = occlusion
```
