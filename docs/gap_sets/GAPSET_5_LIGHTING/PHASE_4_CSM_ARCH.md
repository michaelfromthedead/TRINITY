# Phase 4: Cascaded Shadow Maps Architecture

## Current Status: PARTIAL (0/6 tasks real, 2 partial, 4 absent)

`shadow_csm.wgsl` (161 lines) implements cascade selection and PCF sampling. `shadow.vert.wgsl` + `shadow.frag.wgsl` implement depth-only rendering. No Rust `csm.rs`. No cascade blend. No atlas. No multi-viewport dispatch.

## Architecture

### Pipeline Overview

```
CPU (csm.rs)                              GPU (WGSL)
  |                                          |
  +-- Compute split distances                |
  |   (Python _compute_cascade_splits)       |
  +-- Fit frustum corners                    |
  |   (Python _get_frustum_corners)          |
  +-- Compute cascade matrices               |
  |   (Python _compute_cascade_matrices)     |
  +-- Texel snapping for stabilization       |
  |   (Python _stabilize_bounds)             |
  +-- Upload CascadeData[4] to uniform       |
  |                                          |
  +-- Dispatch shadow.vert/frag              |
  |   for each cascade (or multi-viewport)   |
  |                                          +-- Render depth to cascade layer
  |                                          |
  +-- Lighting pass reads CascadeData        +-- shadow_csm.wgsl functions
```

### Rust Dispatch (csm.rs)

The dispatch must:
1. Compute 4 cascade view-projection matrices
2. Apply texel snapping for temporal stability
3. Render depth for each cascade (4 draw calls or multi-viewport)
4. Upload CascadeData to uniform buffer

Implementation strategies for cascade rendering:
1. **Sequential**: 4 render passes, one per cascade. Simple but 4x draw call overhead.
2. **Multi-viewport**: Single pass with `SetViewport`. Supported by wgpu but limited cascade count.
3. **Array layers**: Single pass writing to 4 array layers via `gl_Layer` in vertex/geometry shader. Most efficient. WGSL `@builtin(vertex_index)` can select layer.

### Cascade Split Computation

Must match Python `CascadedShadowMap._compute_cascade_splits()`:
- Lambda blend between linear and logarithmic
- Default lambda = 0.75
- Default distances: [10, 30, 100, 500] for 4 cascades
- Supports 1-4 cascades
- near=0.1, far up to 1000

### Frustum Fitting

- Compute 8 frustum corners in world space using inverse(view * proj)
- Compute light-space bounding box from transformed corners
- Create orthographic projection from bounding box
- Apply texel snapping: `floor(v / texelSize) * texelSize` to prevent shimmer

### Cascade Selection in Shader

Current `select_cascade()` (shadow_csm.wgsl) chooses the first cascade where `view_depth < split_depth`. This is correct but creates hard seams.

**Fix: Add cascade blend range**
- Blend region: `cascade_split - blend_range/2` to `cascade_split + blend_range/2`
- In blend region, sample both cascades and interpolate
- Default blend range: 2.0 world units (from Python `cascade_blend_range`)

### CSM Atlas vs. Array Textures

**Current approach**: `texture_depth_2d_array` with 4 layers. Clean, but doesn't use atlas.

**Architecture recommendation**: Keep array textures for CSM (only 4 layers, simple addressing). Use shadow atlas (Phase 6) only for point/spot shadows. The 2x2 atlas quadrant approach adds unnecessary complexity for CSM.

### Test Plan

- Split distances match Python: near=0.1, far=1000, lambda=0.75, N=4
- Frustum corners correct: render known points, verify coverage
- Texel snap: compare with/without stabilization on camera motion
- Cascade blend: render gradient, check no visible seam
- Multi-viewport: 4 distinct depth layers produced per frame
