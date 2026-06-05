# Phase 7: Facial & Procedural Animation -- Architecture

## Status: 8 [x] 0 [~] 1 [-]

## Modules: `engine/animation/facial/`, `engine/animation/procedural/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| facial/blend_shapes.py | 724 | Morph target system with ARKit support |
| facial/facs.py | 749 | 52 FACS Action Units with expressions |
| facial/lip_sync.py | 903 | Phoneme/viseme lip synchronization |
| facial/eye_animation.py | 721 | Saccades, blink, pupil, gaze |
| facial/face_rig.py | 756 | Priority-based facial animation layering |
| facial/face_capture.py | 978 | Keyframe capture playback and retargeting |
| procedural/spring_bone.py | 652 | Damped spring physics for secondary motion |
| procedural/lookat.py | 646 | Head/eye look-at with saccades |
| procedural/twist.py | 496 | Twist distribution across bone chains |
| procedural/ragdoll.py | 808 | Physics-driven ragdoll with blending |
| procedural/locomotion.py | 675 | Procedural gait generation |
| procedural/breathing.py | 476 | Natural breathing cycles |
| procedural/secondary_motion.py | 719 | Delay, oscillation, noise, impulse effects |

### Facial Architecture

**Blend Shapes** (`blend_shapes.py`):
- `ARKIT_BLEND_SHAPES`: 52 standard ARKit shape definitions
- `BlendShape`: vertex delta array with weight
- `BlendShapeController`: per-shape weight management
- `CorrectiveBlendShape`: combination shape with trigger weights
- `create_arkit_compatible_set()`: standard ARKit configuration

**FACS** (`facs.py`):
- `ActionUnit`/`AU`: 52 AUs with intensity 0-1, left/right asymmetry
- `Expression`: predefined emotion combinations (anger, joy, surprise, etc.)
- `FACSController`: AU weight to blend shape weight conversion

**Lip Sync** (`lip_sync.py`):
- `Viseme`: 15+ viseme shapes with blend targets
- `PhonemeEvent`: timing-specific phoneme with viseme mapping
- `PHONEME_TO_VISEME`: complete IPA-to-viseme mapping
- `LipSyncController`: coarticulation smoothing, timing shaping

**Eye Animation** (`eye_animation.py`):
- `BlinkController`: random interval (2-6s), duration (100-400ms)
- `EyeController`: gaze with Donders' law, saccades (200-600ms), drift, tremor
- `PupilSettings`: dilation response to light level

### Procedural Architecture

**SpringBone**: damped spring with collision, wind, configurable stiffness/damping/mass
**LookAt**: soft cone limits, weighted chain distribution, velocity prediction
**Twist**: per-bone twist weight distribution along chain
**Ragdoll**: blend-in/out timing, partial activation, active recovery, joint limits
**ProceduralLocomotion**: gait cycle, foot trajectory, body dynamics
**Breathing**: breath rate/depth, exertion level adaptation
**SecondaryMotion**: delay (lagged follow), oscillation, noise (Perlin/simplex), impulse response

### Missing
- T-AN-7.9: Tests

### Key Design Decisions
- ARKit-compatible blend shape naming for industry standard interchange
- FACS asymmetry support enables one-sided expressions (sneer, wink)
- Coarticulation smoothing prevents robotic viseme transitions
- Eye simulation combines 4 motion types (saccade + drift + tremor + blink)
- Spring bones support collision detection (sphere/capsule) for self-intersection prevention
- Look-at uses soft cone limits for natural head movement
- Locomotion is procedural (no animation data required) for non-critical characters
