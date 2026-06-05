# Phase 5: Cube and Spot Shadow Maps Architecture

## Current Status: ABSENT (0/5 tasks real)

No WGSL shaders for cube or spot shadows. Python `CubeShadowMap` and `SpotShadowMap` exist as reference.

## Architecture

### Cube Shadow Maps

For point lights, render 6 faces of a cubemap. Each face is a 90-degree FOV depth pass.

**Cube face layout** (matches Python `CUBE_FACE_DIRECTIONS`):
```
+X:  lookAt(light_pos, light_pos + (1,0,0), (0,-1,0))
-X:  lookAt(light_pos, light_pos + (-1,0,0), (0,-1,0))
+Y:  lookAt(light_pos, light_pos + (0,1,0), (0,0,1))
-Y:  lookAt(light_pos, light_pos + (0,-1,0), (0,0,-1))
+Z:  lookAt(light_pos, light_pos + (0,0,1), (0,-1,0))
-Z:  lookAt(light_pos, light_pos + (0,0,-1), (0,-1,0))
```

**Rendering strategies**:
1. **Cubemap texture** (`texture_depth_cube`): Native cube map, WGSL `textureSampleCompare` with `sampler_comparison` on cubemap. Requires 6 render passes or geometry shader layer selection.
2. **2D array** (`texture_depth_2d_array` with 6 layers): WGSL `textureSampleCompare` on array layer. Simpler to create and bind. Use `@builtin(vertex_index) % 6` or 6 draw calls per face.
3. **Dual-paraboloid**: Single render, 2 hemispheres. Lower quality, cheaper. Not recommended for production.

**Strategy recommendation**: Use 2D array with 6 layers per cube shadow (approach 2). It's the most compatible with wgpu and allows hardware PCF `textureSampleCompare` on each layer.

### Spot Shadow Maps

For spot lights, render a single perspective depth pass matching the spot cone.

**Projection**: `Mat4.perspective(fov=2*outer_angle, aspect=1.0, near=0.1, far=radius)`
- FOV matches the spot light's outer cone angle
- Square aspect ratio (matching shadow map resolution)
- Near plane at 0.1
- Far plane at light's influence radius

### Per-Light Dispatch (cube.rs, spot.rs)

```
cube.rs:
  for each point_light with shadows enabled:
    for face in 0..6:
      set viewport (atlas tile)
      bind face view-projection
      draw depth

spot.rs:
  for each spot_light with shadows enabled:
    set viewport (atlas tile)
    bind spot view-projection
    draw depth
```

### Cone Penumbra Fade

At the spot light's outer cone angle, the shadow should fade smoothly to prevent hard cutoff:

```
shadow_fade = smoothstep(cos(outer_angle), cos(inner_angle), dot(normalize(surface_to_light), light_direction))
shadow = lerp(0.0, pcf_shadow, shadow_fade)
```

This matches Python `SpotLight.get_angular_attenuation()` which uses `t * t * (3 - 2*t)` (smoothstep) between cos_outer and cos_inner.

### Atlas Integration

Both cube and spot shadows render into the shadow atlas (Phase 6). Each light gets an atlas tile via `ShadowAtlas.allocate()`. The per-light viewport is set to the tile's (x, y, width, height).

### Test Plan

- Cube shadow: all 6 faces rendered, depth values in [0,1], faces cover 360 degrees
- Spot shadow: depth within cone frustum, valid distribution
- Cone fade: smooth transition at boundary (no hard cutoff)
- Per-light dispatch: multiple point lights render into correct atlas tiles
