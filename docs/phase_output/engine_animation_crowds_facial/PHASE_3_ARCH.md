# PHASE 3 ARCH: Facial Animation System

**Phase:** 3 of 3
**Focus:** Complete facial animation with FACS, lip sync, and procedural eyes
**Status:** IMPLEMENTED (Investigation confirms REAL)

---

## Phase Overview

Phase 3 implements the complete facial animation pipeline, from low-level blend shapes through high-level emotion control. The system supports industry-standard FACS, ARKit compatibility, and phoneme-based lip sync with coarticulation.

---

## Architecture Components

### 3.1 Blend Shape System

**Module:** `engine/animation/facial/blend_shapes.py`
**Lines:** 724

```
BlendShape
    +-- name: str
    +-- vertex_indices: np.ndarray[int32]
    +-- deltas: np.ndarray[float32, (N, 3)]
    +-- normal_deltas: Optional[np.ndarray]
    +-- tangent_deltas: Optional[np.ndarray]
    
BlendShapeSet
    +-- shapes: dict[str, BlendShape]
    +-- get(name) -> BlendShape
    +-- names() -> list[str]
    
BlendShapeController
    +-- set: BlendShapeSet
    +-- weights: dict[str, float]
    +-- set_weight(name, weight)
    +-- evaluate() -> dict[str, float]
```

**Corrective Shapes:**
```
CorrectiveBlendShape
    +-- driver_shapes: list[str]
    +-- activation_mode: Literal["multiply", "min", "add"]
    +-- correction: BlendShape
```

### 3.2 FACS Implementation

**Module:** `engine/animation/facial/facs.py`
**Lines:** 749

```
ActionUnit (Enum):
    AU1_INNER_BROW_RAISER      # Frontalis (medial)
    AU2_OUTER_BROW_RAISER      # Frontalis (lateral)
    AU4_BROW_LOWERER           # Corrugator supercilii
    AU6_CHEEK_RAISER           # Orbicularis oculi (orbital)
    AU12_LIP_CORNER_PULLER     # Zygomaticus major
    AU14_DIMPLER               # Buccinator
    AU43_EYES_CLOSED           # Relaxation of Levator palpebrae
    ... (21 total)

Expression (Enum):
    NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, DISGUSTED, FEARFUL, CONTEMPT
    
ExpressionData
    +-- expression: Expression
    +-- au_weights: dict[ActionUnit, float]       # Bilateral
    +-- au_left_weights: dict[ActionUnit, float]  # Left only
    +-- au_right_weights: dict[ActionUnit, float] # Right only
    
FACSController
    +-- set_au_weight(au, weight, side="both")
    +-- set_expression(expression, intensity)
    +-- get_blend_weights() -> dict[str, float]
```

**Anatomical Correctness:**
- CONTEMPT is asymmetric (unilateral lip corner)
- Each AU maps to specific facial muscles
- Bilateral support for asymmetric expressions

### 3.3 Lip Sync System

**Module:** `engine/animation/facial/lip_sync.py`
**Lines:** 903

```
Viseme (Enum):
    SILENCE, AA, EE, IH, OH, OO, ...  # 15 visemes

PhonemeEvent
    +-- phoneme: str (IPA or ARPABET)
    +-- start_time: float
    +-- end_time: float
    +-- weight: float
    
VisemeEvent
    +-- viseme: Viseme
    +-- start_time: float
    +-- end_time: float
    +-- weight: float
    
LipSyncController
    +-- events: list[VisemeEvent]
    +-- intensity: float
    +-- blend_time: float
    +-- update(time) -> dict[str, float]
```

**Coarticulation Algorithm (Lines 446-509):**
```
apply_coarticulation(events, settings) -> list[VisemeEvent]:
    for i, event in enumerate(events):
        # Anticipation: blend toward next phoneme early
        if i < len(events) - 1:
            next_event = events[i + 1]
            anticipation_weight = settings.anticipation_strength
            event.blend_with(next_event, anticipation_weight, at_end=True)
        
        # Carryover: previous phoneme persists
        if i > 0:
            prev_event = events[i - 1]
            carryover_weight = settings.carryover_strength
            event.blend_with(prev_event, carryover_weight, at_start=True)
    
    return events
```

**Phoneme-to-Viseme Mapping:**
- 60+ IPA/ARPABET phonemes
- Maps to 15 Preston Blair-derived visemes
- ARKit-compatible blend shape names

### 3.4 Eye Animation System

**Module:** `engine/animation/facial/eye_animation.py`
**Lines:** 721

```
EyeTransform
    +-- rotation: Quaternion
    +-- vergence: float
    +-- pupil_dilation: float
    
EyeController
    +-- left_eye: EyeTransform
    +-- right_eye: EyeTransform
    +-- look_at(target: Vec3)
    +-- update(dt)
```

**Procedural Behaviors:**

| Behavior | Implementation | Parameters |
|----------|---------------|------------|
| Look-At | Target tracking | target, speed |
| Vergence | Eye convergence | eye_separation, distance |
| Micro-Saccades | Random movements | amplitude, frequency |
| Blinking | Automatic blinks | interval_range, duration |
| Pupil Dilation | Light/emotion response | light_level, arousal |

**Vergence Formula (Lines 598-622):**
```python
vergence_angle = degrees(atan2(eye_separation * 0.5, distance))
vergence_angle = min(vergence_angle, max_vergence)
left_eye.vergence = vergence_angle      # turns right (positive)
right_eye.vergence = -vergence_angle    # turns left (negative)
```

### 3.5 Face Capture System

**Module:** `engine/animation/facial/face_capture.py`
**Lines:** 978

```
AnimationCurve
    +-- keyframes: list[Keyframe]
    +-- interpolation: Literal["linear", "step", "cubic", "hermite"]
    +-- evaluate(time) -> float
    
FaceCaptureClip
    +-- duration: float
    +-- curves: dict[str, AnimationCurve]  # shape_name -> curve
    +-- sample(time) -> dict[str, float]
    
FaceCapturePlayer
    +-- clip: FaceCaptureClip
    +-- speed: float
    +-- loop: bool
    +-- update(dt) -> dict[str, float]
    
FaceCaptureRetargeter
    +-- source_to_target: dict[str, list[tuple[str, float, float]]]
    +-- retarget(weights) -> dict[str, float]
```

### 3.6 Face Rig Integration

**Module:** `engine/animation/facial/face_rig.py`
**Lines:** 756

```
LayerPriority (Enum):
    IDLE = 0
    EMOTION = 1
    LIP_SYNC = 2
    PROCEDURAL = 3  # Eyes
    OVERRIDE = 4
    
AnimationLayer
    +-- priority: LayerPriority
    +-- weight: float
    +-- is_additive: bool
    +-- weights: dict[str, float]
    
FaceRig
    +-- blend_controller: BlendShapeController
    +-- facs_controller: FACSController
    +-- lip_sync: LipSyncController
    +-- eye_controller: EyeController
    +-- layers: dict[str, AnimationLayer]
    +-- update(dt) -> dict[str, float]
```

**Layer Blending (Lines 594-630):**
```python
def _blend_layers(self) -> dict[str, float]:
    result = {}
    sorted_layers = sorted(self._layers.values(), key=lambda l: l.priority.value)
    
    for layer in sorted_layers:
        for shape_name, value in layer.weights.items():
            weighted_value = value * layer.weight
            
            if layer.is_additive:
                result[shape_name] = result.get(shape_name, 0.0) + weighted_value
            else:
                # Override blend
                base = result.get(shape_name, 0.0)
                result[shape_name] = base * (1.0 - layer.weight) + weighted_value
    
    return result
```

---

## Dependencies

### Internal
- `engine/animation/config.py` - `FacialConfig`
- `engine/core/math` - `Vec3`, `Quaternion`
- NumPy for vectorized operations

### External
- None (pure Python)

---

## Interfaces

### Input Interface (ARKit Integration)
```python
class ARKitFaceData:
    blend_shapes: dict[str, float]  # 52 ARKit blend shapes
    head_transform: Transform
    eye_transforms: tuple[Transform, Transform]
```

### Output Interface
```python
class FaceRigOutput:
    blend_weights: dict[str, float]
    eye_rotations: tuple[Quaternion, Quaternion]
    jaw_rotation: Quaternion
```

---

## Quality Attributes

### Correctness
- FACS based on Ekman research
- Anatomically accurate AU-to-muscle mapping
- Coarticulation for natural speech

### Performance
- Sparse blend shapes for memory efficiency
- NumPy vectorization for batch operations
- Priority layers avoid redundant blending

### Compatibility
- ARKit 52 blend shape names
- IPA/ARPABET phoneme support
- Preston Blair viseme set

---

## Verification Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| All 8 Ekman expressions render correctly | Visual comparison |
| Coarticulation produces smooth speech | Audio-visual sync test |
| Eye vergence converges at near targets | Geometric verification |
| Priority layers blend correctly | Weight assertion tests |
| ARKit shapes map 1:1 | Name enumeration test |
