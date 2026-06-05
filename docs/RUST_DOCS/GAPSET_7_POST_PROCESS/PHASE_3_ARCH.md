# Phase 3: Cinematic Effects -- Architecture

**Phase ID**: PHASE_3 | **Tasks**: 13 (4 [x], 6 [~], 3 [-])

---

## Module Structure

### Python: `dof.py` (652 lines)

```
DOFSettings (dataclass)
  - focal_length, aperture, sensor_width, focus_distance
  - max_blur_radius (default 32), quality_level, auto_focus

DOFEffect (PostProcessEffect)
  - execute() -> stub (calls blur methods that return None)

CircleOfConvergence
  - calculate(depth, focal_length, aperture, sensor_width, focus_distance) -> CoC
  - Formula: abs(f^2 * (depth - focus_dist) / (aperture * depth * (focus_dist - f)))
  - Hyperfocal distance: H = f^2 / (aperture * sensor_width)
  - near_field_coeff, far_field_coeff
  - Returns CoC in pixels, sign indicates foreground(-)/background(+)
  - CPU working: correct physical model

BokehShape
  - ShapeType: CIRCLE, POLYGON, ANAMORPHIC, CAT_EYE, SWIRL
  - generate_samples(shape_type, blade_count, angle, intensity_profile)
  - Disk: uniform sampling within radius
  - Polygon: blade_count vertices (5/6/7/8/9), sample within polygon
  - Anamorphic: elliptical with aspect ratio
  - Cat eye: shape clipped by aperture
  - CPU reference: sample point generation works

NearFieldDOF
  - dilate_coc() -> stub (returns None)
  - blur(near_field, coc) -> stub (returns None)

FarFieldDOF
  - blur(far_field, coc) -> stub (returns None)

AutoFocusSystem
  - focus_distance, focus_speed, autofocus_mode (CENTER, FACIAL, CUSTOM)
  - update(depth_buffer) -> adjusts focus distance
  - CPU working: correct tracking logic

BokehSettings (dataclass)
  - shape, blade_count, max_blur, anamorphic_ratio, cat_eye_intensity
  - vortex_intensity, vortex_radius (SWIRL mode)
```

### Python: `motion_blur.py` (550 lines)

```
MotionBlurSettings (dataclass)
  - sample_count (default 16), tile_size, shutter_angle
  - camera_blur, object_blur, half_resolution, denoise

MotionBlurEffect (PostProcessEffect)
  - execute() -> stub (calls apply that returns input)

CameraMotionBlur
  - apply_blur(color, depth, velocity, camera) -> stub (returns input)
  - _project_to_screen(world_pos, view_proj) -> CPU matrix multiply
  - _unproject_from_screen(uv, depth, inv_view_proj)
  - CPU architecture: correct geometric model for camera motion

ObjectMotionBlur
  - apply_blur(color, velocity, depth, settings) -> stub (returns input)
  - velocity_from_delta(previous_uv, current_uv) -> motion vector
  - CPU architecture: correct velocity computation

TileMaxVelocity
  - calculate_tile_max(velocity_buffer, tile_size) -> stub (returns None)
  - calculate_neighbor_max(tile_max, dilation) -> stub (returns None)
  - CPU architecture: correct tile-based optimization structure

CombinedMotionBlur
  - apply_combined(color, depth, velocity, camera_motion, settings) -> stub
```

---

## Data Flow

```
HDR Scene Color (after bloom)
  |
  v
[DOF: Circle of Confusion] -- per-pixel CoC calculation
  |
  v
[DOF: Tile Max CoC] -- reduce CoC to tiles (STUB)
  |
  v
[DOF: Near Field Blur] -- foreground defocus (STUB)
  |
  v
[DOF: Far Field Blur] -- background defocus (STUB)
  |
  v
[DOF: Composite] -- lerp(scene, blurred, CoC/max_CoC) (STUB)
  |
  v
[Motion Blur: Velocity] -- camera + object velocity (STUB)
  |
  v
[Motion Blur: Tile Max] -- velocity reduction (STUB)
  |
  v
[Motion Blur: Sampling] -- per-pixel blur (STUB)
  |
  v
[Motion Blur: Denoise] -- bilateral filter (STUB)
  |
  v
To AO / Next effect
```

**Key observation**: Both DOF and motion blur have complete architectural structures with correct physical models (CoC formula, bokeh shapes, velocity reprojection, tile max). All execute steps are stubs that don't produce pixel output.

---

## What Exists ([x] = 4)

| Task | Component | State |
|------|-----------|-------|
| T-PP-3.1 | DOF Circle of Confusion | Working: physical CoC formula, hyperfocal distance, CPU reference |
| T-PP-3.1b | Quality CoC radius control | Complete: max_blur_radius 4/8/16/32, quality presets set values |
| T-PP-3.2a | Bokeh shape variants | Complete: 5 shapes (CIRCLE, POLYGON, ANAMORPHIC, CAT_EYE, SWIRL) |
| T-PP-3.3 | Motion blur velocity | Complete: tile max, camera, object motion blur classes |

## What Is Partial ([~] = 6)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-3.1a | Tile max CoC reduction | dilate_coc() exists but returns None |
| T-PP-3.2 | DOF bokeh gather | BokehShape generates samples, blur() methods return None, no composite |
| T-PP-3.2b | FG/BG separation | Intensity multipliers defined, no composite logic |
| T-PP-3.3a | Rotated interleaved sampling | Sample count configurable, no interleaved pattern |
| T-PP-3.3b | Velocity resolution by quality | half_resolution flag exists, no quality resolution mapping |
| T-PP-3.4 | Motion blur bilateral denoise | BilateralFilter exists (in AO module), not connected to motion blur |

## What Is Missing ([-] = 3)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-3.5 | Film grain | No implementation anywhere |
| T-PP-3.5a | Chrominance grain quality | No implementation |
| T-PP-3.6 | Phase 3 integration tests | None |

---

## Bokeh Shape Detail

| Shape | Sample Generation | Quality Level |
|-------|------------------|---------------|
| CIRCLE | Uniform random within radius | Low |
| POLYGON | Uniform random within N-gon (5-9 blades) | Medium/High |
| ANAMORPHIC | Elliptical with aspect ratio | Ultra |
| CAT_EYE | Polygon clipped by circular aperture | Ultra |
| SWIRL | Polygon with angular vortex | Artistic |

All implementations are CPU Python reference only.

---

## Interfaces

### CoC Calculation Contract (implemented, CPU)

```python
# Input: depth buffer value, camera parameters
# Output: CoC in pixels with sign (foreground < 0, background > 0)
def calculate(depth, focal_length, aperture, sensor_width, focus_distance):
    hyperfocal = focal_length**2 / (aperture * sensor_width)
    coc = abs(focal_length**2 * (depth - focus_distance) /
              (aperture * depth * (focus_distance - focal_length)))
    coc_pixels = coc * sensor_width
    return coc_pixels * sign(depth - focus_distance)
```

### Velocity Buffer Contract (planned, not implemented)

```rust
// Input: current UV + previous frame UV
// Output: motion vector in UV space (2x float)
struct VelocityBuffer {
    velocity: vec2<f32>,  // UV space displacement
}
```

### Bilateral Filter Contract (implemented, CPU)

```python
# Depth-aware spatial denoise
weight = spatial_gaussian(dx, dy, sigma) 
       * exp(-depth_diff^2 * sharpness) 
       * max(0, dot(normal_a, normal_b))
```
