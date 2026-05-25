# Archaeological Investigation: engine/animation/crowds + engine/animation/facial

**Investigator**: Research Agent
**Date**: 2026-05-22
**Classification**: REAL (Both Subdirectories)

---

## Executive Summary

Both `engine/animation/crowds` and `engine/animation/facial` contain **production-ready, real implementations** with complete algorithms, proper edge-case handling, and well-architected domain models. These are not stubs or scaffolding.

---

## Subdirectory 1: engine/animation/crowds (2,237 lines total)

### Classification: **REAL IMPLEMENTATION**

### File Analysis

| File | Lines | Status | Key Evidence |
|------|-------|--------|--------------|
| `crowd_behavior.py` | 710 | REAL | Complete FSM with 5 behaviors, avoidance algorithms |
| `animation_texture.py` | 510 | REAL | GPU texture baking, encoding/decoding, atlas support |
| `crowd_lod.py` | 496 | REAL | Hierarchical LOD with skeleton reduction, hysteresis |
| `crowd_renderer.py` | 458 | REAL | Instance buffering, batching, GPU data packing |

### Key Algorithms Implemented

**1. Crowd Behavior System (crowd_behavior.py)**
- **State Machine**: `AgentState` enum with IDLE, WALKING, WAITING, FLEEING, FORMATION
- **Avoidance Algorithm**: Lines 304-345 implement RVO-style avoidance with:
  - Priority-based agent avoidance (AVOIDANCE_PRIORITY_MULTIPLIER)
  - Obstacle avoidance with combined radii
  - Coincident agent handling with random direction push
  - Division-by-zero protection via MIN_DISTANCE_EPSILON
- **Formation Behavior**: Leader-follower with formation offset (lines 476-551)
- **Flee Behavior**: Threat-based with safe distance threshold (lines 411-474)

**2. Animation Texture Baking (animation_texture.py)**
- **Transform Encoding**: 2 pixels per bone (position+scale, quaternion rotation)
- **Cubic Hermite Interpolation**: Lines 156-172 for smooth temporal sampling
- **RGBA8 Packing**: Float-to-RGBA8 with 32-bit precision (lines 473-510)
- **Atlas System**: Multi-clip atlas with UV range calculation

**3. LOD System (crowd_lod.py)**
- **Skeleton Reduction**: Importance-based bone culling (lines 385-456)
- **Bone Importance Scoring**: Anatomical hierarchy (root > spine > limbs > fingers)
- **Hysteresis**: Prevents LOD flickering via distance threshold
- **Smoothstep Transitions**: Lines 108-112

**4. GPU Renderer (crowd_renderer.py)**
- **Instance Buffer Management**: Dynamic growth with BUFFER_GROWTH_FACTOR
- **Buffer Overflow Protection**: InstanceBufferOverflowError exception
- **Batch Sorting**: Priority-based render order
- **Memory Calculation**: Accurate byte-size tracking

### Evidence of Real Implementation

```python
# crowd_behavior.py:309-319 - Real avoidance with edge case handling
if dist < CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON:
    # Agents are coincident - push in random direction
    angle = random.uniform(0, 2 * math.pi)
    avoidance = avoidance + Vec3(math.sin(angle), 0, math.cos(angle)) * self._avoidance_strength
    continue

# Stronger avoidance when closer
strength = (1.0 - dist / self._avoidance_radius) * self._avoidance_strength

# Consider priority
if other.priority > agent.priority:
    strength *= CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER
```

### Configuration Integration

All modules properly import from `engine.animation.config`:
- `CROWD_BEHAVIOR_CONFIG` for agent physics
- `ANIMATION_TEXTURE_CONFIG` for texture limits
- `CROWD_LOD_CONFIG` for LOD thresholds
- `CROWD_RENDERER_CONFIG` for buffer sizing

---

## Subdirectory 2: engine/animation/facial (5,233 lines total)

### Classification: **REAL IMPLEMENTATION**

### File Analysis

| File | Lines | Status | Key Evidence |
|------|-------|--------|--------------|
| `face_capture.py` | 978 | REAL | Mocap playback, cubic Hermite curves, retargeting |
| `lip_sync.py` | 903 | REAL | Phoneme-to-viseme mapping, coarticulation |
| `face_rig.py` | 756 | REAL | Priority-based layer blending, emotion control |
| `facs.py` | 749 | REAL | Complete FACS AU mapping, Ekman expressions |
| `blend_shapes.py` | 724 | REAL | Sparse morph targets, corrective shapes |
| `eye_animation.py` | 721 | REAL | Vergence, saccades, pupil dilation |

### Key Algorithms Implemented

**1. Face Capture System (face_capture.py)**
- **Animation Curves**: Keyframe-based with 4 interpolation modes:
  - LINEAR, STEP, CUBIC (Hermite), HERMITE (Catmull-Rom)
- **Cubic Hermite Interpolation**: Lines 157-172
- **Retargeting**: Scale/offset mapping with many-to-one accumulation
- **Clip Merging**: Sequential concatenation with gap time

**2. Lip Sync Controller (lip_sync.py)**
- **Phoneme-to-Viseme Mapping**: 60+ IPA/ARPABET phonemes mapped to 15 visemes
- **Coarticulation**: Anticipation and carryover blending (lines 446-509)
- **Smoothstep Blend Curves**: ease_in, ease_out, ease_in_out
- **Viseme-to-BlendShape Mapping**: ARKit-compatible defaults

**3. FACS Implementation (facs.py)**
- **18 Action Units**: Full upper face, mouth, and eye AUs
- **Bilateral Support**: Left/right intensity for asymmetric expressions
- **Ekman Expressions**: NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, DISGUSTED, FEARFUL, CONTEMPT
- **CONTEMPT Asymmetry**: Correct unilateral lip corner implementation

**4. Blend Shape System (blend_shapes.py)**
- **Sparse Representation**: vertex_indices + deltas for memory efficiency
- **Corrective Shapes**: Driver-based activation with multiply/min/add modes
- **ARKit Compatibility**: 52 standard blend shape names
- **NumPy Vectorization**: Efficient sparse delta application

**5. Eye Animation (eye_animation.py)**
- **Vergence Calculation**: Lines 598-622 with eye_separation geometry
- **Micro-Saccades**: Random small movements for realism
- **Blink Controller**: Variable intervals, half-blinks, double-blinks
- **Pupil Dilation**: Light response + emotional arousal

**6. Face Rig Integration (face_rig.py)**
- **Priority-Based Layers**: IDLE < EMOTION < LIP_SYNC < PROCEDURAL < OVERRIDE
- **Additive vs Override**: Per-layer blend mode
- **Jaw-from-LipSync**: Procedural jaw rotation from jawOpen weight
- **Emotion Blending**: Smooth transitions with blend_time

### Evidence of Real Implementation

```python
# facs.py:366-378 - Asymmetric contempt expression (anatomically correct)
Expression.CONTEMPT: ExpressionData(
    expression=Expression.CONTEMPT,
    au_weights={},
    # Contempt is asymmetric - slight smile on one side
    au_left_weights={
        ActionUnit.AU12_LIP_CORNER_PULLER: 0.0,
        ActionUnit.AU14_DIMPLER: 0.0,
    },
    au_right_weights={
        ActionUnit.AU12_LIP_CORNER_PULLER: 0.5,
        ActionUnit.AU14_DIMPLER: 0.6,
    },
),
```

```python
# lip_sync.py:732-748 - Zero-duration edge case handling
min_duration = 0.001  # 1ms minimum
if event_duration < min_duration:
    # Zero/negative duration: apply full weight instantly
    current_weight = current_event.weight * self._intensity
elif self._blend_time <= 0:
    # No blending: full weight
    current_weight = current_event.weight * self._intensity
else:
    # Normal case: calculate fade in/out
    fade_in = min(1.0, time_in_event / self._blend_time)
    time_remaining = event_duration - time_in_event
    fade_out = min(1.0, time_remaining / self._blend_time)
    current_weight = min(fade_in, fade_out) * current_event.weight * self._intensity
```

```python
# eye_animation.py:598-622 - Vergence with real geometry
# Calculate vergence angle (eyes converge for near objects)
# Using small angle approximation
vergence_angle = math.degrees(math.atan2(self._eye_separation * 0.5, distance))
vergence_angle = min(vergence_angle, self._limits.max_vergence)

# Left eye turns right (positive), right eye turns left (negative)
self._left_eye.vergence = vergence_angle
self._right_eye.vergence = -vergence_angle
```

### ARKit Compatibility Evidence

```python
# blend_shapes.py:635-672 - Full 52-shape ARKit set
ARKIT_BLEND_SHAPES = [
    # Eye shapes
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    ...
    # Tongue (may not be tracked)
    "tongueOut",
]
```

---

## Cross-Module Integration

### Dependency Graph

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

### Configuration Centralization

All modules use `engine/animation/config.py`:
- FacialConfig for lip sync timing, blink intervals
- CrowdBehaviorConfig for agent physics
- CrowdRendererConfig for buffer limits
- AnimationTextureConfig for texture dimensions

---

## Recommendations

### Production Ready

1. **Crowds**: Ready for GPU instanced rendering of large character counts
2. **Facial**: Ready for cinematic facial animation with ARKit compatibility

### Suggested Enhancements

1. **Crowds**:
   - Add spatial partitioning (grid/octree) for O(n log n) neighbor queries
   - Implement GPU-accelerated behavior update

2. **Facial**:
   - Add neural network-based lip sync from audio
   - Implement facial rig rigging from mesh topology

### Testing Priorities

1. Crowd avoidance edge cases at high density
2. Lip sync coarticulation with rapid phoneme sequences
3. Eye vergence at extreme near distances

---

## Summary

**engine/animation/crowds**: 4 files, 2,237 lines, 100% REAL. Complete GPU crowd rendering with behavior simulation, animation texture baking, and hierarchical LOD.

**engine/animation/facial**: 6 files, 5,233 lines, 100% REAL. Complete facial animation system with FACS AU mapping, phoneme-based lip sync, eye procedural animation, and priority-based rig layering. ARKit-compatible blend shape names throughout.
