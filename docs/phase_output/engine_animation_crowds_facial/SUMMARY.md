# SUMMARY: engine/animation/crowds + engine/animation/facial

---

## Metrics

| Metric | Crowds | Facial | Total |
|--------|--------|--------|-------|
| Files | 4 | 6 | 10 |
| Lines | 2,237 | 5,233 | 7,470 |
| Classification | REAL | REAL | REAL |
| Production Ready | YES | YES | YES |

### File Breakdown

| Subsystem | File | Lines |
|-----------|------|-------|
| Crowds | `crowd_behavior.py` | 710 |
| Crowds | `animation_texture.py` | 510 |
| Crowds | `crowd_lod.py` | 496 |
| Crowds | `crowd_renderer.py` | 458 |
| Facial | `face_capture.py` | 978 |
| Facial | `lip_sync.py` | 903 |
| Facial | `face_rig.py` | 756 |
| Facial | `facs.py` | 749 |
| Facial | `blend_shapes.py` | 724 |
| Facial | `eye_animation.py` | 721 |

---

## Algorithm Inventory

### Crowds Algorithms

| Algorithm | File | Lines | Status | Description |
|-----------|------|-------|--------|-------------|
| Agent State Machine | crowd_behavior.py | 304-345 | COMPLETE | 5-state FSM (IDLE, WALKING, WAITING, FLEEING, FORMATION) |
| RVO-Style Avoidance | crowd_behavior.py | 304-345 | COMPLETE | Priority-based avoidance with obstacle handling |
| Formation Behavior | crowd_behavior.py | 476-551 | COMPLETE | Leader-follower with formation offset |
| Flee Behavior | crowd_behavior.py | 411-474 | COMPLETE | Threat-based with safe distance threshold |
| Cubic Hermite Interpolation | animation_texture.py | 156-172 | COMPLETE | Smooth temporal sampling for animation |
| RGBA8 Packing | animation_texture.py | 473-510 | COMPLETE | Float-to-RGBA8 with 32-bit precision |
| Animation Atlas | animation_texture.py | - | COMPLETE | Multi-clip atlas with UV range calculation |
| Skeleton Reduction | crowd_lod.py | 385-456 | COMPLETE | Importance-based bone culling |
| Bone Importance Scoring | crowd_lod.py | - | COMPLETE | Anatomical hierarchy (root > spine > limbs > fingers) |
| Hysteresis LOD | crowd_lod.py | - | COMPLETE | Prevents LOD flickering |
| Smoothstep Transitions | crowd_lod.py | 108-112 | COMPLETE | Smooth LOD blending |
| Instance Buffer Management | crowd_renderer.py | - | COMPLETE | Dynamic growth with overflow protection |
| Batch Sorting | crowd_renderer.py | - | COMPLETE | Priority-based render order |

### Facial Algorithms

| Algorithm | File | Lines | Status | Description |
|-----------|------|-------|--------|-------------|
| Animation Curves | face_capture.py | 157-172 | COMPLETE | 4 interpolation modes (LINEAR, STEP, CUBIC, HERMITE) |
| Retargeting | face_capture.py | - | COMPLETE | Scale/offset mapping with many-to-one accumulation |
| Clip Merging | face_capture.py | - | COMPLETE | Sequential concatenation with gap time |
| Phoneme-to-Viseme | lip_sync.py | - | COMPLETE | 60+ IPA/ARPABET phonemes to 15 visemes |
| Coarticulation | lip_sync.py | 446-509 | COMPLETE | Anticipation and carryover blending |
| Smoothstep Blend | lip_sync.py | - | COMPLETE | ease_in, ease_out, ease_in_out |
| Zero-Duration Handling | lip_sync.py | 732-748 | COMPLETE | Edge case for instant phonemes |
| FACS Action Units | facs.py | - | COMPLETE | 18 AUs with bilateral support |
| Ekman Expressions | facs.py | 366-378 | COMPLETE | 8 expressions including asymmetric CONTEMPT |
| Sparse Blend Shapes | blend_shapes.py | - | COMPLETE | vertex_indices + deltas for memory efficiency |
| Corrective Shapes | blend_shapes.py | - | COMPLETE | Driver-based activation (multiply/min/add) |
| ARKit Compatibility | blend_shapes.py | 635-672 | COMPLETE | 52 standard blend shape names |
| Eye Vergence | eye_animation.py | 598-622 | COMPLETE | Geometry-based convergence for near objects |
| Micro-Saccades | eye_animation.py | - | COMPLETE | Random small movements for realism |
| Blink Controller | eye_animation.py | - | COMPLETE | Variable intervals, half-blinks, double-blinks |
| Pupil Dilation | eye_animation.py | - | COMPLETE | Light response + emotional arousal |
| Priority Layer Blending | face_rig.py | - | COMPLETE | IDLE < EMOTION < LIP_SYNC < PROCEDURAL < OVERRIDE |
| Jaw-from-LipSync | face_rig.py | - | COMPLETE | Procedural jaw rotation from jawOpen weight |

---

## Evidence Snippets

### RVO-Style Avoidance with Edge Cases

crowd_behavior.py:309-319:
- Coincident agents pushed in random direction
- Strength inversely proportional to distance
- Priority-based avoidance multiplier

### Asymmetric CONTEMPT Expression

facs.py:366-378:
- CONTEMPT is anatomically asymmetric (unilateral lip corner)
- au_left_weights vs au_right_weights for bilateral control

### Zero-Duration Phoneme Handling

lip_sync.py:732-748:
- 1ms minimum duration threshold
- Instant weight application for zero-duration events
- Proper fade_in/fade_out calculation otherwise

### Eye Vergence Geometry

eye_animation.py:598-622:
- Vergence angle from atan2(eye_separation/2, distance)
- Max vergence clamping
- Left eye positive, right eye negative

---

## Configuration Integration

| Config | Used By | Purpose |
|--------|---------|---------|
| CROWD_BEHAVIOR_CONFIG | crowd_behavior.py | Agent physics, avoidance parameters |
| ANIMATION_TEXTURE_CONFIG | animation_texture.py | Texture dimensions, precision |
| CROWD_LOD_CONFIG | crowd_lod.py | LOD thresholds, bone importance |
| CROWD_RENDERER_CONFIG | crowd_renderer.py | Buffer sizing, batch limits |
| FacialConfig | facial modules | Lip sync timing, blink intervals |
