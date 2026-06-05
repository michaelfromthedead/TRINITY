# Investigation: engine/animation/facial

## Summary
The facial animation module is a **comprehensive, production-quality implementation** totaling 5,233 lines across 8 files. It provides complete systems for blend shapes with sparse vertex deltas, FACS-based anatomical expressions mapped to ARKit's 52 blend shapes, phoneme-to-viseme lip sync with coarticulation, procedural eye animation with vergence/saccades/blinking, motion capture playback with retargeting, and a unified face rig controller with priority-based animation layering.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 141 | REAL | Clean exports for all 40+ public classes/functions |
| `blend_shapes.py` | 724 | REAL | Full BlendShape, BlendShapeSet, BlendShapeController with correctives |
| `facs.py` | 749 | REAL | Complete FACS with 21 Action Units, 8 expression presets |
| `lip_sync.py` | 903 | REAL | Full viseme system with coarticulation and phoneme mapping |
| `eye_animation.py` | 721 | REAL | EyeController with vergence, saccades, blinks, pupil dilation |
| `face_rig.py` | 756 | REAL | Unified FaceRig integrating all subsystems |
| `face_capture.py` | 978 | REAL | AnimationCurve, FaceCaptureClip, Player, Retargeter |
| `config.py` | 261 | REAL | Comprehensive configuration dataclasses |

## Facial Components
- **Blend Shapes**: Sparse vertex delta representation with corrective blend shapes
- **FACS**: 21 Action Units (AU1-AU43) with bilateral support, 8 Ekman emotions
- **Lip Sync**: 15 visemes, coarticulation with anticipation/carryover, phoneme-to-viseme mapping
- **Eye Animation**: Look-at tracking, micro-saccades, vergence, automatic blinking with variation
- **Pupil Response**: Light and emotional arousal response
- **Face Capture**: Keyframe curves (linear/cubic/hermite), clip playback, retargeting
- **Face Rig**: Priority-based animation layers (idle, emotion, lip_sync, eyes, override)

## Implementation
- Real blendshapes? **YES** - Full sparse vertex delta system with numpy arrays, normal/tangent deltas, corrective shapes
- Real FACS? **YES** - Complete Ekman FACS with AU1-AU43, bilateral asymmetry support, expression blending
- Real lip sync? **YES** - Full phoneme-to-viseme pipeline with coarticulation, timeline playback, ARKit compatibility

## Verdict
**REAL IMPLEMENTATION** - This is production-quality facial animation code with:
- Professional sparse blend shape representation
- Anatomically accurate FACS based on Ekman's research
- Industry-standard viseme set (Preston Blair phoneme chart)
- Procedural eye behavior matching human physiology
- ARKit 52 blend shape compatibility for iOS face tracking
- Motion capture retargeting pipeline

## Evidence

### Blend Shapes - Sparse Vertex Deltas (blend_shapes.py:30-108)
```python
@dataclass
class BlendShape:
    name: str
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    deltas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32).reshape(0, 3))
    normal_deltas: Optional[np.ndarray] = None
    tangent_deltas: Optional[np.ndarray] = None
```

### FACS Action Units (facs.py:20-57)
```python
class ActionUnit(Enum):
    AU1_INNER_BROW_RAISER = auto()    # Frontalis (medial)
    AU2_OUTER_BROW_RAISER = auto()    # Frontalis (lateral)
    AU4_BROW_LOWERER = auto()         # Corrugator supercilii, Depressor supercilii
    AU6_CHEEK_RAISER = auto()         # Orbicularis oculi (orbital)
    AU12_LIP_CORNER_PULLER = auto()   # Zygomaticus major
    # ... 21 total AUs with anatomical muscle annotations
```

### Lip Sync Coarticulation (lip_sync.py:446-509)
```python
def apply_coarticulation(events: list[VisemeEvent], settings: CoarticulationSettings):
    # Calculate influence from previous phoneme (carryover)
    # Calculate influence from next phoneme (anticipation)
    # Returns blended viseme states with smooth transitions
```

### Eye Vergence Calculation (eye_animation.py:598-622)
```python
def _update_vergence(self) -> None:
    # Calculate vergence angle (eyes converge for near objects)
    vergence_angle = math.degrees(math.atan2(self._eye_separation * 0.5, distance))
    # Left eye turns right (positive), right eye turns left (negative)
    self._left_eye.vergence = vergence_angle
    self._right_eye.vergence = -vergence_angle
```

### Face Rig Layer Blending (face_rig.py:594-630)
```python
def _blend_layers(self) -> dict[str, float]:
    sorted_layers = sorted(self._layers.values(), key=lambda l: l.priority.value)
    for layer in sorted_layers:
        if layer.is_additive:
            result[shape_name] = result.get(shape_name, 0.0) + weighted_value
        else:
            result[shape_name] = result[shape_name] * (1.0 - layer.weight) + weighted_value
```

### ARKit 52 Compatibility (blend_shapes.py:635-672)
```python
ARKIT_BLEND_SHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "mouthSmileLeft", "mouthSmileRight",
    # ... all 52 standard ARKit blend shapes
]
```
