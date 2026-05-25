# PHASE 1 ARCH: Core GPU Crowd Rendering Pipeline

**Phase:** 1 of 3
**Focus:** GPU-accelerated crowd rendering with animation textures
**Status:** IMPLEMENTED (Investigation confirms REAL)

---

## Phase Overview

Phase 1 establishes the foundational GPU crowd rendering pipeline, enabling thousands of animated characters through texture-based skeletal animation and instanced rendering.

---

## Architecture Components

### 1.1 Animation Texture System

**Module:** `engine/animation/crowds/animation_texture.py`
**Lines:** 511

```
AnimationTexture
    +-- bone_count: int
    +-- frame_count: int  
    +-- duration: float
    +-- texture_data: np.ndarray (RGBA8)
    +-- sample_bone_transform(bone_index, time) -> Transform
    
AnimationTextureAtlas
    +-- textures: dict[str, AnimationTexture]
    +-- pack() -> combined texture + UV ranges
```

**Key Algorithms:**
- Transform encoding: 2 pixels per bone (position+scale, quaternion)
- Cubic Hermite interpolation for smooth sampling
- Float-to-RGBA8 packing with 32-bit precision

**Data Flow:**
```
AnimationClip -> bake_clip_to_texture() -> AnimationTexture -> Atlas
                                              |
                                              v
                                        GPU Texture Upload
```

### 1.2 Instance Buffer Management

**Module:** `engine/animation/crowds/crowd_renderer.py`
**Lines:** 459

```
InstanceBuffer
    +-- transform_data: np.ndarray (float32, 16 per instance)
    +-- animation_data: np.ndarray (float32, 4 per instance)
    +-- color_data: np.ndarray (float32, 4 per instance)
    +-- add_instance(CrowdInstance) -> index
    +-- clear()
    +-- get_gpu_data() -> packed bytes
```

**Memory Layout:**
| Component | Elements | Bytes/Instance |
|-----------|----------|----------------|
| Transform | 16 floats (4x4 matrix) | 64 |
| Animation | 4 floats (clip, time, speed, LOD) | 16 |
| Color | 4 floats (RGBA tint) | 16 |
| **Total** | 24 floats | **96 bytes** |

### 1.3 Batch Rendering

**Module:** `engine/animation/crowds/crowd_renderer.py`

```
CrowdRenderBatch
    +-- mesh_id: int
    +-- material_id: int
    +-- instances: list[CrowdInstance]
    +-- priority: int (render order)
    
CrowdRenderer
    +-- batches: dict[(mesh, material), CrowdRenderBatch]
    +-- atlases: dict[str, AnimationTextureAtlas]
    +-- add_instance(instance)
    +-- render() -> GPU commands
```

**Batching Strategy:**
1. Group by mesh + material
2. Sort batches by priority
3. Upload instance buffers per batch
4. Single draw call per batch

---

## Dependencies

### Internal
- `engine/animation/config.py` - `CROWD_RENDERER_CONFIG`
- `engine/core/math` - `Vec3`, `Quaternion`, `Transform`

### External
- NumPy for array operations
- GPU backend for actual rendering (Rust)

---

## Interfaces

### Input Interface
```python
class CrowdInstance:
    position: Vec3
    rotation: Quaternion
    scale: Vec3
    animation_index: int
    animation_time: float
    animation_speed: float
    tint_color: Vec4
    lod_level: int
```

### Output Interface
```python
class GPURenderCommand:
    mesh_id: int
    material_id: int
    instance_count: int
    transform_buffer: bytes
    animation_buffer: bytes
    color_buffer: bytes
    texture_atlas: int
```

---

## Quality Attributes

### Performance
- Target: 10,000+ instances at 60fps
- Single draw call per batch
- GPU-side animation sampling

### Reliability
- Buffer overflow protection (`InstanceBufferOverflowError`)
- Dynamic buffer growth (`BUFFER_GROWTH_FACTOR`)
- Validation on texture baking

### Maintainability
- Configuration-driven parameters
- Clear separation: baking / buffering / batching
- Unit testable components

---

## Verification Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Animation textures encode correctly | Round-trip encode/decode test |
| Instance buffers pack correctly | Memory layout verification |
| Batching groups correctly | Batch count assertions |
| Buffer overflow handled | Exception test at capacity |
