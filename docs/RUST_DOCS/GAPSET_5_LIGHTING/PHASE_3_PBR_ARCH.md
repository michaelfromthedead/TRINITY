# Phase 3: PBR Lighting Architecture

## Current Status: PARTIAL (1/7 tasks real, 6 absent)

`pbr.frag.wgsl` (377 lines) implements forward PBR with Cook-Torrance BRDF for 3 light types. No deferred compute pass exists. No area light, IES, or sky light evaluation in WGSL.

## Architecture Decision: Forward+ Over Deferred

The TODO describes a deferred compute pipeline (`lighting_pass.comp.wgsl`). The existing codebase implements forward PBR (`pbr.frag.wgsl`). **Recommendation: Target Forward+ (forward with froxel-culled light lists).**

Rationale:
1. `pbr.frag.wgsl` already implements correct Cook-Torrance BRDF
2. Froxel culling from Phase 2 can feed per-tile light lists
3. No G-buffer, no render target management, no extra bandwidth
4. Forward rendering handles MSAA, alpha blending, and transparent objects naturally

### Forward+ Shader Architecture

```
pbr.vert.wgsl  --  pbr.frag.wgsl (modified)
                       |
                       +---> froxel_light_list (storage buffer from Phase 2)
                       +---> direct light iteration for directional + sky
                       +---> BRDF evaluation for each froxel-assigned light
```

The fragment shader would:
1. Compute fragment screen position
2. Compute froxel index from (tile_x, tile_y, depth_slice)
3. Read light_offset + light_count from froxel_grid
4. Iterate only lights assigned to this froxel
5. Evaluate BRDF for each light

### Light Evaluation Functions

**Already exists** (in `pbr.frag.wgsl`):
- `eval_directional_light()` -- correct implementation
- `eval_point_light()` -- correct implementation (attenuation: `pow(clamp(1 - d^2/r^2, 0, 1), 2)`)
- `eval_spot_light()` -- correct implementation (distance att + cone att via smoothstep)
- `eval_brdf()` -- Cook-Torrance with GGX NDF, Smith-GGX geometry, Schlick Fresnel

**Needs to be built**:
- `eval_rect_area_light()` -- requires LTC LUT textures (see T-LIT-3.3/3.4)
- `eval_disk_area_light()` -- requires LTC LUT (shared with rect)
- `eval_ies_light()` -- requires IES texture sampler (see T-LIT-3.5)
- `eval_sky_light()` -- sample cubemap or SH probe

### Shared Light Eval Module

Instead of `light_eval.wgsl` (which doesn't exist), the BRDF functions should remain in `pbr.frag.wgsl` or be extracted to a shared `brdf.wgsl` include. The Python reference for attenuation functions (`PointLight.get_attenuation()`, `SpotLight.get_angular_attenuation()`) should be the gold standard.

### LTC for Area Lights (T-LIT-3.3, 3.4)

Linearly Transformed Cosines require:
- Precomputed LUT: 64x64 RGBA16F (inverse M matrix) + 64x64 R16F (magnitude)
- Generate via `ltc_lut.py` or a compute shader
- Sampled in WGSL with texture_2d<f32>
- Implementation follows Heitz 2016 "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"

### IES Light Evaluation (T-LIT-3.5)

- Load IES profile as 2D texture (horizontal_angle x vertical_angle)
- Sample in WGSL with bilinear filtering
- Multiply by distance attenuation

### Test Plan

Compare WGSL output against Python reference:
- 100 random configurations per light type (position, direction, intensity, color, roughness, metallic)
- 10 test scenes for full pass validation
- Pixel-level comparison with per-component tolerance
