# PHASE 1 TODO: Shadow Map Infrastructure

## T-LGT-1.1: Shadow Texture Factory

**Description:** Create factory function to allocate shadow depth textures via renderer-backend.

**Tasks:**
- [ ] Add `ShadowTextureFactory` class in `engine/rendering/lighting/gpu/shadow_textures.py`
- [ ] Implement `create_cascade_texture(cascade_count, resolution)` returning array texture
- [ ] Implement `create_cube_texture(resolution)` returning cubemap depth texture
- [ ] Implement `create_spot_texture(resolution)` returning 2D depth texture
- [ ] Wire factory to renderer-backend's `TextureTable.allocate()`
- [ ] Add comparison sampler creation with clamp-to-border (depth=1.0)

**Acceptance Criteria:**
- [ ] `create_cascade_texture(4, 2048)` returns valid `DepthTextureArray` with 4 layers
- [ ] `create_cube_texture(1024)` returns valid `DepthCubeTexture` with 6 faces
- [ ] Textures appear in renderer-backend's resource tracking
- [ ] Unit test confirms texture dimensions and format

---

## T-LGT-1.2: Replace Placeholder Handles in shadows.py

**Description:** Replace `_texture_handle: int = 0` with actual GPU resource references.

**Tasks:**
- [ ] Add `_depth_texture: Optional[DepthTexture]` field to `CascadedShadowMap`
- [ ] Add `_depth_texture: Optional[DepthCubeTexture]` field to `CubeShadowMap`
- [ ] Add `_depth_texture: Optional[DepthTexture]` field to `SpotShadowMap`
- [ ] Modify constructors to optionally accept pre-allocated textures
- [ ] Add `get_texture()` method returning the GPU resource
- [ ] Add `get_sampler()` method returning comparison sampler
- [ ] Deprecate `_texture_handle` and `_depth_handle` integer fields

**Acceptance Criteria:**
- [ ] `csm.get_texture()` returns non-None `DepthTextureArray`
- [ ] `cube.get_texture()` returns non-None `DepthCubeTexture`
- [ ] Existing CPU-only tests still pass (texture=None path preserved)
- [ ] No integer handles remain in public API

---

## T-LGT-1.3: Shadow Atlas GPU Integration

**Description:** Wire `ShadowAtlas` to allocate from a real GPU depth texture.

**Tasks:**
- [ ] Add `_atlas_texture: DepthTexture` field to `ShadowAtlas`
- [ ] Modify `__init__` to create atlas texture with configurable size
- [ ] Modify `allocate()` to return `ShadowAtlasRegion` with UV coordinates
- [ ] Add `get_uv_transform(region_id)` returning `(offset, scale)` tuple
- [ ] Add `get_texture()` method for shader binding
- [ ] Implement `defragment()` to compact allocations (texture copy required)

**Acceptance Criteria:**
- [ ] `atlas.allocate(512, 512)` returns region with valid UV in [0,1] range
- [ ] Multiple allocations return non-overlapping UV regions
- [ ] `get_texture()` returns the shared atlas depth texture
- [ ] Defragmentation moves allocations without visual artifacts

---

## T-LGT-1.4: CSM View-Projection Matrix Upload

**Description:** Ensure cascade matrices are available for GPU shaders.

**Tasks:**
- [ ] Add `CascadeUniforms` struct: `view_proj: Mat4`, `split_near: f32`, `split_far: f32`
- [ ] Add `get_cascade_uniforms(cascade_index)` method to `CascadedShadowMap`
- [ ] Add `get_all_cascade_uniforms()` returning array of all cascades
- [ ] Format uniforms for GPU buffer upload (16-byte aligned)

**Acceptance Criteria:**
- [ ] `get_cascade_uniforms(0)` returns first cascade's view-projection
- [ ] Matrices match existing `_compute_view_matrix()` output
- [ ] Split values match `_compute_cascade_splits()` output
- [ ] Unit test verifies frustum coverage for all cascades

---

## T-LGT-1.5: Shadow Render Pass Registration

**Description:** Register shadow map rendering with frame graph.

**Tasks:**
- [ ] Create `ShadowRenderPass` class in `engine/rendering/lighting/gpu/shadow_pass.py`
- [ ] Implement `register_csm_passes(frame_graph, csm, scene)` for N cascade passes
- [ ] Implement `register_cube_pass(frame_graph, cube_shadow, scene)` for 6-face render
- [ ] Implement `register_spot_pass(frame_graph, spot_shadow, scene)` for single pass
- [ ] Configure depth-only output, LESS comparison, front-face culling
- [ ] Add geometry culling per cascade frustum

**Acceptance Criteria:**
- [ ] Frame graph shows N passes for N-cascade CSM
- [ ] Each pass renders only geometry within its cascade frustum
- [ ] Cube shadow uses 6 passes (or 1 with geometry shader if available)
- [ ] Shadow atlas shows non-empty depth values after render

---

## T-LGT-1.6: Shadow Bias Configuration

**Description:** Expose shadow bias parameters for artifact reduction.

**Tasks:**
- [ ] Add `ShadowBias` dataclass: `constant: f32`, `slope_scale: f32`, `normal_offset: f32`
- [ ] Add `bias` field to all shadow map classes
- [ ] Apply constant bias in depth shader
- [ ] Apply slope-scale bias based on surface angle
- [ ] Expose per-light bias override in `DirectionalLight`, `SpotLight`, `PointLight`

**Acceptance Criteria:**
- [ ] Shadow acne eliminated with default bias values
- [ ] Peter-panning minimized (shadows stay attached to objects)
- [ ] Bias values configurable per light for fine-tuning
- [ ] Unit test validates bias application in shader output

---

## T-LGT-1.7: Shadow Map Debug Visualization

**Description:** Add debug views for shadow map inspection.

**Tasks:**
- [ ] Add `render_shadow_debug(shadow_map, viewport)` function
- [ ] Implement cascade visualization (color-coded by cascade index)
- [ ] Implement atlas utilization view (packed regions outlined)
- [ ] Implement depth buffer visualization (linear depth to color)
- [ ] Wire to debug UI toggle

**Acceptance Criteria:**
- [ ] Debug view shows all 4 cascades with distinct colors
- [ ] Atlas view shows allocated regions and free space
- [ ] Depth visualization reveals shadow map content
- [ ] No performance impact when debug disabled
