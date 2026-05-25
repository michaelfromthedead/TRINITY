# Phase 8: Crowd System -- Architecture

## Status: 3 [x] 0 [~] 3 [-]

## Module: `engine/animation/crowds/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| animation_texture.py | 510 | Bone transform baking to RGBA textures |
| crowd_renderer.py | 458 | GPU instanced rendering with per-instance data |
| crowd_lod.py | 496 | LOD selection and reduced skeleton |
| crowd_behavior.py | 710 | Agent steering and behavior simulation |
| __init__.py | 64 | Public API exports |

### Architecture

**AnimationTextures** (`animation_texture.py`):
- `TextureFormat`: RGBA8, RGBA16F, RGBA32F
- `AnimationTexture`: bone transform texture (bone_index x clip_frame)
- `AnimationTextureAtlas`: multi-clip texture atlas
- `bake_clip_to_texture()`: transforms clip into per-frame RGBA pixel data
- `encode_transform_to_pixels()`: Vec3/Quat -> 4 RGBA values
- `decode_pixels_to_transform()`: RGBA values -> Vec3/Quat
- Config from `config.py`: max 4096x4096, default 1024x2048, 256 bones max

**CrowdRenderer** (`crowd_renderer.py`):
- `CrowdInstance`: per-instance data (position, animation_id, phase, lod, tint)
- `CrowdRenderBatch`: grouped instances sharing mesh/texture
- `InstanceBuffer`: GPU buffer management with dynamic resizing
- `CrowdRenderer`: draw call management, frustum culling, LOD sorting

**CrowdLOD** (`crowd_lod.py`):
- `LODLevel`: FULL_SKELETON (0), SIMPLIFIED (1), IMPOSTOR (2)
- `CrowdLOD`: distance-based selection with hysteresis
- `LODTransition`: dither/fade between LOD levels
- `create_reduced_skeleton()`: importance-based bone reduction (4 bone minimum)

**CrowdBehavior** (`crowd_behavior.py`):
- `AgentState`: IDLE, WALKING, WAITING, FLEEING, FORMATION
- `CrowdAgent`: position, velocity, target, state, personality traits
- Behavior types: IDLE (variation 3-8s), WALKING (1.4 m/s), WAITING, FLEEING (1.5x speed), FORMATION
- `CrowdBehavior`: per-agent behavior state machine
- `CrowdSimulator`: world-aware agent update with collision avoidance

### Missing
- T-AN-8.3: `shaders/crowd/crowd_skinning.vert.wgsl` -- vertex skinning shader
- T-AN-8.4: `shaders/crowd/impostor.frag.wgsl` -- impostor fragment shader
- T-AN-8.6: Tests

### Key Design Decisions
- Animation textures use RGBA8 encoding with range packing for GPU efficiency
- Renderer uses GPU instancing: single draw call per crowd mesh
- LOD operates at 3 levels: full skeleton, 4-bone simplified, impostor (planned)
- Agent behaviors use state machine (6 states) with priority-based transition
- Simulation supports up to 100K agents with 10K visible
- Animation texture atlas supports multiple clips per character class
- Python crowd system prepares data structures for future WGSL shader consumption
