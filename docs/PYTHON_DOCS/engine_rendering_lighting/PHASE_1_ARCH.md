# PHASE 1 ARCHITECTURE: Shadow Map Infrastructure

## Overview

Phase 1 creates real GPU resources for shadow maps. This is the foundation for all other lighting GPU integration - shadows must work before filtering, and filtering must work before GI can use shadow visibility.

## Components

### 1.1 Shadow Texture Creation

Replace placeholder integer handles with actual GPU depth textures.

**Current State (shadows.py:72-74):**
```python
_texture_handle: int = 0
_depth_handle: int = 0
```

**Target State:**
```python
_depth_texture: DepthTexture  # From renderer-backend
_sampler: ShadowSampler       # Comparison sampler
```

**Architecture Decision:** Use the renderer-backend's `TextureTable` for shadow texture allocation. Shadow maps are depth-only textures with comparison sampling enabled.

### 1.2 Cascaded Shadow Map Textures

CSM requires array textures (one layer per cascade).

**Specification:**
- Format: `D32Float` (32-bit depth)
- Dimensions: Configurable (default 2048x2048 per cascade)
- Array layers: 1-4 based on `cascade_count`
- Sampler: Comparison mode, clamp-to-border with depth=1.0

**Integration Point:** Frame graph must schedule N depth passes (one per cascade) with proper frustum culling per cascade.

### 1.3 Cube Shadow Map Textures

Point light shadows render to cubemaps.

**Current State (shadows.py):**
- 6 face matrices computed correctly
- No actual cubemap texture

**Target State:**
- `DepthCubeTexture` with 6 faces
- Single-pass or multi-pass rendering (geometry shader optional)
- Face indexing matches existing direction vectors

### 1.4 Shadow Atlas

The atlas packer (`ShadowAtlas`) already implements best-fit rectangle packing. It needs to:

1. Allocate from a real GPU depth texture
2. Return valid UV regions for sampling
3. Track used regions for defragmentation

**Architecture Decision:** Single large depth texture (default 4096x4096) subdivided by the atlas allocator. This reduces texture binding overhead.

**Data Flow:**
```
ShadowAtlas.allocate(width, height)
  -> PackedRect with (x, y, width, height)
  -> UV transform for shader: (x/atlas_width, y/atlas_height, width/atlas_width, height/atlas_height)
```

### 1.5 Shadow Render Passes

Each shadow map requires a depth-only render pass.

**Frame Graph Integration:**
```
for each shadow_caster in scene:
    fg.add_pass(
        name=f"shadow_{caster.id}",
        output=shadow_atlas.region_for(caster),
        depth_func=LESS,
        cull_mode=FRONT,  # Reduce peter-panning
        geometry=caster.shadow_casters
    )
```

**Optimization:** Sort shadow casters by atlas region to minimize render target switches.

## Data Structures

### ShadowMapResource
```
struct ShadowMapResource {
    texture: DepthTexture,
    sampler: Sampler,
    view_proj: Mat4,
    cascade_matrices: Option<[Mat4; 4]>,
    cascade_splits: Option<[f32; 4]>,
}
```

### ShadowAtlasRegion
```
struct ShadowAtlasRegion {
    uv_offset: Vec2,
    uv_scale: Vec2,
    resolution: u32,
    shadow_type: enum { Directional, Spot, Point },
}
```

## Shader Interface

Phase 1 produces shadow maps; Phase 2 consumes them. The interface:

```wgsl
struct ShadowData {
    view_proj: mat4x4<f32>,
    cascade_splits: vec4<f32>,  // For CSM
    atlas_region: vec4<f32>,    // xy=offset, zw=scale
    bias: vec2<f32>,            // constant, slope-scale
    filter_size: f32,
    shadow_type: u32,           // 0=dir, 1=spot, 2=point
}
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Atlas fragmentation | Defrag on frame boundary when utilization < 70% |
| CSM cascade seams | Overlap cascades by 10% for blend zone |
| Cube shadow overdraw | Use geometry shader if supported, else 6 passes |
| Memory exhaustion | Pool shadow maps, evict LRU for off-screen lights |

## Dependencies

- `renderer-backend::TextureTable` for texture allocation
- `renderer-backend::FrameGraph` for pass scheduling
- Existing frustum culling for cascade geometry selection
