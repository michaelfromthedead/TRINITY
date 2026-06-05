# MASTER: engine_animation_crowds_facial

**RDC Consolidated Knowledge Document**
**Generated:** 2026-05-23
**Status:** REAL IMPLEMENTATION (100%)

---

## 1. Subsystem Overview

The crowds and facial animation subsystems represent **7,470 lines of production-quality Python code** implementing GPU-accelerated crowd rendering, behavior simulation, and comprehensive facial animation with industry-standard compatibility.

### Classification
- **Crowds**: REAL IMPLEMENTATION (2,237 lines)
- **Facial**: REAL IMPLEMENTATION (5,233 lines)
- **Combined Status**: Production-ready

---

## 2. Crowds Subsystem Architecture

### 2.1 Animation Texture System

**Component:** `animation_texture.py` (511 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| AnimationTexture | Baked animation clips as GPU textures | Position+scale+rotation per bone per frame |
| AnimationTextureAtlas | Multiple clips packed into single texture | UV range calculation for multi-clip |
| Transform Encoding | 2 pixels per bone encoding scheme | Position+scale in pixel 1, quaternion in pixel 2 |
| RGBA8 Packing | Float-to-RGBA8 with 32-bit precision | Lines 473-510 |
| Cubic Hermite Interpolation | Smooth temporal sampling | Lines 156-172 |
| Skeleton Data Structure | Bone hierarchy for animation | Hierarchical bone data |
| AnimationClip | Clip data for baking | Keyframe sequences |

**Key Functions:**
- `bake_clip_to_texture()`: Full baking pipeline with validation
- `encode_transform_to_pixels()` / `decode_pixels_to_transform()`: Transform codec
- `sample_bone_transform()`: Arbitrary time sampling with interpolation

### 2.2 Crowd Behavior System

**Component:** `crowd_behavior.py` (711 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| CrowdAgent | Full agent state model | Position, velocity, facing, priority, animations, group |
| AgentState | State machine enum | IDLE, WALKING, WAITING, FLEEING, FORMATION |
| CrowdBehavior | Abstract behavior base | Extensible behavior interface |
| CrowdSimulator | Main simulation loop | Agent and behavior management |
| BehaviorContext | Shared agent context | Nearby agents, obstacles |

**Behavior Types:**
1. `IdleBehavior`: Standing with animation variation
2. `WalkingBehavior`: Movement with steering avoidance
3. `WaitingBehavior`: Queue-like waiting with fidgeting
4. `FleeingBehavior`: Panic flee from threat source
5. `FormationBehavior`: Leader-follower with formation offset

**Avoidance Algorithm (Lines 304-345):**
- RVO-style avoidance with priority-based weighting
- `AVOIDANCE_PRIORITY_MULTIPLIER` for priority agents
- Obstacle avoidance with combined radii
- Coincident agent handling with random direction push
- Division-by-zero protection via `MIN_DISTANCE_EPSILON`

### 2.3 LOD System

**Component:** `crowd_lod.py` (497 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| LODLevel | Per-level configuration | Distance, bone count, update rate, shadows |
| CrowdLOD | LOD manager | Hysteresis-based level selection |
| LODTransition | Transition modes | Instant, blend, dither |
| Skeleton Reduction | Intelligent simplification | Importance-based bone culling |
| Bone Importance Scoring | Anatomical hierarchy | Root > spine > limbs > fingers |
| Smoothstep Transitions | Flicker prevention | Lines 108-112 |

**Bone Importance Hierarchy:**
- Root/pelvis: 0.95 importance
- Spine/head: 0.85-0.90 importance
- Arms/legs: 0.75 importance
- Hands/feet: 0.60 importance
- Fingers/toes: 0.40 importance
- Twist/helper bones: -0.20 penalty

### 2.4 GPU Renderer

**Component:** `crowd_renderer.py` (459 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| CrowdInstance | Per-instance data | Transform, animation, tint, LOD |
| CrowdRenderer | Main renderer | Batch and atlas management |
| CrowdRenderBatch | Instance grouping | By mesh/material |
| InstanceBuffer | GPU-ready buffer | Transform/animation/color packing |
| Buffer Overflow Protection | Safety mechanism | `InstanceBufferOverflowError` exception |
| Dynamic Buffer Growth | Capacity management | `BUFFER_GROWTH_FACTOR` |

---

## 3. Facial Subsystem Architecture

### 3.1 Blend Shape System

**Component:** `blend_shapes.py` (724 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| BlendShape | Sparse morph target | vertex_indices + deltas |
| BlendShapeSet | Shape collection | Named shape lookup |
| BlendShapeController | Weight management | Evaluation pipeline |
| Corrective Shapes | Combination corrections | Driver-based activation |
| Activation Modes | Corrective triggers | Multiply, min, add |
| ARKit 52 Shapes | Industry standard | Full compatibility |
| NumPy Vectorization | Performance | Efficient sparse delta application |

**ARKit Compatibility:**
- 52 standard blend shape names
- Eye shapes (blink, look directions)
- Mouth shapes (smile, frown, pucker)
- Tongue tracking support

### 3.2 FACS Implementation

**Component:** `facs.py` (749 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| ActionUnit | FACS AU definition | 21 AUs with anatomical annotations |
| FACSController | AU weight management | Bilateral support |
| Expression | Ekman emotion presets | 8 base expressions |
| ExpressionData | AU weight mappings | Per-expression configurations |
| Bilateral Support | Left/right intensity | Asymmetric expressions |
| Anatomical Accuracy | Muscle-based AUs | Ekman research compliance |

**Action Units (21 total):**
- AU1: Inner Brow Raiser (Frontalis medial)
- AU2: Outer Brow Raiser (Frontalis lateral)
- AU4: Brow Lowerer (Corrugator supercilii)
- AU6: Cheek Raiser (Orbicularis oculi orbital)
- AU12: Lip Corner Puller (Zygomaticus major)
- AU14: Dimpler
- AU43: Eyes Closed
- ... and 14 more

**Ekman Expressions (8):**
1. NEUTRAL
2. HAPPY
3. SAD
4. ANGRY
5. SURPRISED
6. DISGUSTED
7. FEARFUL
8. CONTEMPT (asymmetric - unilateral lip corner)

### 3.3 Lip Sync System

**Component:** `lip_sync.py` (903 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| LipSyncController | Main controller | Timeline-based playback |
| PhonemeEvent | Phoneme timing | IPA/ARPABET support |
| VisemeEvent | Visual speech shapes | 15 viseme set |
| Coarticulation | Transition blending | Anticipation + carryover |
| CoarticulationSettings | Blend parameters | Timing configurations |
| Phoneme-to-Viseme Map | Sound-to-shape mapping | 60+ phonemes to 15 visemes |
| Smoothstep Blend Curves | Transition smoothing | ease_in, ease_out, ease_in_out |

**Viseme Set (15):**
- Preston Blair phoneme chart derived
- ARKit-compatible shape names
- Includes silence/rest shapes

**Coarticulation Implementation (Lines 446-509):**
- Anticipation: Next phoneme influence
- Carryover: Previous phoneme persistence
- Smooth transitions via blending

**Edge Case Handling (Lines 732-748):**
- Zero-duration phonemes
- No-blend instant transitions
- Fade in/out calculation

### 3.4 Eye Animation System

**Component:** `eye_animation.py` (721 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| EyeController | Main eye controller | Dual eye management |
| EyeTransform | Per-eye state | Rotation, vergence, dilation |
| Look-At Tracking | Gaze direction | Target-based orientation |
| Vergence Calculation | Eye convergence | Geometry-based (Lines 598-622) |
| Micro-Saccades | Subtle movements | Random small adjustments |
| Blink Controller | Automatic blinking | Variable intervals |
| Blink Variations | Natural variety | Half-blinks, double-blinks |
| Pupil Dilation | Size response | Light + emotional arousal |

**Vergence Formula:**
```
vergence_angle = degrees(atan2(eye_separation * 0.5, distance))
left_eye.vergence = vergence_angle      # turns right
right_eye.vergence = -vergence_angle    # turns left
```

### 3.5 Face Capture System

**Component:** `face_capture.py` (978 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| AnimationCurve | Keyframe curves | 4 interpolation modes |
| FaceCaptureClip | Captured data | Timeline of shape weights |
| FaceCapturePlayer | Playback engine | Speed, looping, events |
| FaceCaptureRetargeter | Shape mapping | Source-to-target conversion |
| Interpolation Modes | Curve smoothing | LINEAR, STEP, CUBIC, HERMITE |
| Retargeting | Scale/offset mapping | Many-to-one accumulation |
| Clip Merging | Sequential concatenation | Gap time handling |

**Interpolation Types:**
- LINEAR: Direct lerp
- STEP: Instant transitions
- CUBIC: Hermite spline (Lines 157-172)
- HERMITE: Catmull-Rom variant

### 3.6 Face Rig Integration

**Component:** `face_rig.py` (756 lines)

| Concept | Description | Implementation |
|---------|-------------|----------------|
| FaceRig | Unified controller | All subsystem integration |
| AnimationLayer | Priority layers | Weighted blend stacks |
| LayerPriority | Priority enum | IDLE < EMOTION < LIP_SYNC < PROCEDURAL < OVERRIDE |
| Blend Modes | Layer combination | Additive vs Override |
| Jaw-from-LipSync | Procedural jaw | jawOpen weight to rotation |
| Emotion Blending | Smooth transitions | blend_time parameter |

**Layer Priority Order:**
1. IDLE (lowest)
2. EMOTION
3. LIP_SYNC
4. PROCEDURAL (eyes)
5. OVERRIDE (highest)

---

## 4. Cross-Module Dependencies

### 4.1 Dependency Graph

```
face_rig.py
    +-- blend_shapes.py (BlendShapeController, BlendShapeSet)
    +-- eye_animation.py (EyeController, EyeTransform)
    +-- facs.py (ActionUnit, Expression, FACSController)
    +-- lip_sync.py (LipSyncController, PhonemeEvent, VisemeEvent)

crowd_renderer.py
    +-- animation_texture.py (AnimationTexture, AnimationTextureAtlas)

crowd_lod.py
    +-- animation_texture.py (Skeleton)
```

### 4.2 Configuration Integration

All modules use `engine/animation/config.py`:
- `FacialConfig`: Lip sync timing, blink intervals
- `CrowdBehaviorConfig`: Agent physics parameters
- `CrowdRendererConfig`: Buffer limits
- `AnimationTextureConfig`: Texture dimensions

---

## 5. Key Algorithms Summary

### Physics & Simulation
- RVO-style crowd avoidance with priority weighting
- Steering behaviors with arrival slowdown

### Animation
- Cubic Hermite interpolation for curves
- Animation texture baking for GPU
- Skeleton reduction by importance

### Facial
- FACS Action Unit blending
- Coarticulation for lip sync
- Eye vergence geometry
- Priority-based layer blending

---

## 6. Industry Standards Compliance

| Standard | Implementation |
|----------|---------------|
| ARKit 52 Blend Shapes | Full compatibility |
| FACS (Ekman) | 21 Action Units |
| Preston Blair Visemes | 15 viseme set |
| IPA/ARPABET Phonemes | 60+ phoneme support |

---

## 7. Missing Components (Identified Gaps)

### Crowds
1. **Flow Fields**: No grid-based flow field navigation (uses direct avoidance)
2. **Spatial Partitioning**: `get_nearby_agents()` is O(n) linear scan
3. **GPU Compute Simulation**: Agent updates are CPU-side only
4. **NavMesh Integration**: No pathfinding integration

### Facial
1. **Neural Network Lip Sync**: No audio-to-viseme ML
2. **Auto Rigging**: No mesh topology to rig generation
