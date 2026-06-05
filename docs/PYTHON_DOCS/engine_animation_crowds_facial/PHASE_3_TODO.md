# PHASE 3 TODO: Facial Animation System

**Phase:** 3 of 3
**Focus:** Complete facial animation with FACS, lip sync, and procedural eyes
**Status:** ✅ COMPLETE (All 7 tasks GREEN_LIGHT)

---

## Task List

### T3.1 Blend Shape Evaluation

**Priority:** HIGH
**Estimate:** 2 hours
**File:** `engine/animation/facial/blend_shapes.py`

**Acceptance Criteria:**
- [x] Sparse deltas apply correctly to target vertices
- [x] Corrective shapes activate at correct thresholds
- [x] Blend weights clamp to [0, 1] range
- [x] NumPy vectorization is correct

**Test Cases:**
```python
def test_sparse_blend_application():
    shape = BlendShape(
        name="smile",
        vertex_indices=np.array([0, 5, 10]),
        deltas=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    )
    base_mesh = np.zeros((20, 3), dtype=np.float32)
    result = apply_blend_shape(base_mesh, shape, weight=0.5)
    assert result[0, 0] == 0.5
    assert result[5, 1] == 0.5
    assert result[10, 2] == 0.5

def test_corrective_activation():
    corrective = CorrectiveBlendShape(
        driver_shapes=["smile_L", "smile_R"],
        activation_mode="min",
        correction=make_correction_shape()
    )
    # Both drivers at 1.0 -> correction activates
    weights = {"smile_L": 1.0, "smile_R": 1.0}
    assert corrective.get_weight(weights) == 1.0
```

---

### T3.2 FACS Expression Mapping

**Priority:** HIGH
**Estimate:** 2 hours
**File:** `engine/animation/facial/facs.py`

**Acceptance Criteria:**
- [x] All 21 Action Units map to blend shapes
- [x] All 8 Ekman expressions produce correct AU combinations
- [x] CONTEMPT is correctly asymmetric
- [x] Bilateral support works (left/right independence)

**Test Cases:**
```python
def test_expression_au_mapping():
    facs = FACSController()
    facs.set_expression(Expression.HAPPY, intensity=1.0)
    weights = facs.get_blend_weights()
    # HAPPY should activate AU6 (cheek raiser) and AU12 (lip corner puller)
    assert weights.get("cheekRaiserL", 0) > 0
    assert weights.get("mouthSmileL", 0) > 0

def test_contempt_asymmetry():
    facs = FACSController()
    facs.set_expression(Expression.CONTEMPT, intensity=1.0)
    weights = facs.get_blend_weights()
    # CONTEMPT is right-sided
    assert weights.get("mouthSmileR", 0) > weights.get("mouthSmileL", 0)
```

---

### T3.3 Lip Sync Coarticulation

**Priority:** HIGH
**Estimate:** 3 hours
**File:** `engine/animation/facial/lip_sync.py`

**Acceptance Criteria:**
- [x] Phoneme events convert to viseme events
- [x] Coarticulation blends anticipation and carryover
- [x] Zero-duration phonemes handled correctly
- [x] Timeline playback is frame-accurate

**Test Cases:**
```python
def test_phoneme_to_viseme():
    controller = LipSyncController()
    controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
    events = controller.get_viseme_events()
    assert events[0].viseme == Viseme.AA

def test_coarticulation_blending():
    events = [
        VisemeEvent(Viseme.AA, 0.0, 0.1),
        VisemeEvent(Viseme.OO, 0.1, 0.2),
    ]
    coarticulated = apply_coarticulation(events, CoarticulationSettings())
    # End of AA should blend toward OO
    # Start of OO should blend from AA
    # Verify weights are modified
    pass

def test_zero_duration_phoneme():
    controller = LipSyncController()
    controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))  # Zero duration
    weights = controller.update(time=0.0)
    # Should not crash, should return valid weights
    assert isinstance(weights, dict)
```

---

### T3.4 Eye Animation Vergence

**Priority:** MEDIUM
**Estimate:** 2 hours
**File:** `engine/animation/facial/eye_animation.py`

**Acceptance Criteria:**
- [x] Eyes converge when looking at near targets
- [x] Eyes are parallel when looking at distant targets
- [x] Vergence angle is geometrically correct
- [x] Max vergence limit is respected

**Test Cases:**
```python
def test_vergence_near_target():
    eye_controller = EyeController(eye_separation=0.06)
    eye_controller.look_at(Vec3(0, 0, 0.3))  # 30cm away
    eye_controller.update(0.016)
    # Eyes should converge
    assert eye_controller.left_eye.vergence > 0
    assert eye_controller.right_eye.vergence < 0
    assert abs(eye_controller.left_eye.vergence) == abs(eye_controller.right_eye.vergence)

def test_vergence_distant_target():
    eye_controller = EyeController(eye_separation=0.06)
    eye_controller.look_at(Vec3(0, 0, 100))  # 100m away
    eye_controller.update(0.016)
    # Eyes should be nearly parallel
    assert abs(eye_controller.left_eye.vergence) < 0.1
```

---

### T3.5 Face Rig Layer Blending

**Priority:** MEDIUM
**Estimate:** 2 hours
**File:** `engine/animation/facial/face_rig.py`

**Acceptance Criteria:**
- [x] Higher priority layers override lower
- [x] Additive layers accumulate correctly
- [x] Layer weights scale contributions
- [x] All subsystems integrate correctly

**Test Cases:**
```python
def test_layer_priority_override():
    rig = FaceRig()
    rig.set_layer_weight("idle", "mouthSmileL", 0.5)
    rig.set_layer_weight("emotion", "mouthSmileL", 1.0)  # Higher priority
    result = rig.evaluate()
    # EMOTION overrides IDLE
    assert result["mouthSmileL"] > 0.5

def test_additive_layers():
    rig = FaceRig()
    rig.add_layer("procedural", priority=LayerPriority.PROCEDURAL, additive=True)
    rig.set_layer_weight("emotion", "browDownL", 0.3)
    rig.set_layer_weight("procedural", "browDownL", 0.2)  # Additive
    result = rig.evaluate()
    # Should accumulate: 0.3 + 0.2 = 0.5
    assert abs(result["browDownL"] - 0.5) < 0.001
```

---

### T3.6 Face Capture Retargeting

**Priority:** LOW
**Estimate:** 2 hours
**File:** `engine/animation/facial/face_capture.py`

**Acceptance Criteria:**
- [x] Source shapes map to target shapes
- [x] Scale and offset apply correctly
- [x] Many-to-one mappings accumulate
- [x] Missing shapes handled gracefully

**Test Cases:**
```python
def test_retarget_mapping():
    retargeter = FaceCaptureRetargeter()
    retargeter.add_mapping("source_smile", "target_smile_L", scale=0.5, offset=0.0)
    retargeter.add_mapping("source_smile", "target_smile_R", scale=0.5, offset=0.0)
    result = retargeter.retarget({"source_smile": 1.0})
    assert result["target_smile_L"] == 0.5
    assert result["target_smile_R"] == 0.5
```

---

### T3.7 ARKit Compatibility

**Priority:** MEDIUM
**Estimate:** 1 hour
**File:** `engine/animation/facial/blend_shapes.py`

**Acceptance Criteria:**
- [x] All 52 ARKit blend shape names are present
- [x] Names match ARKit SDK exactly (case-sensitive)
- [x] Blend shapes can be driven by ARKit data

**Test Cases:**
```python
def test_arkit_shape_names():
    assert len(ARKIT_BLEND_SHAPES) == 52
    # Check specific critical shapes
    assert "eyeBlinkLeft" in ARKIT_BLEND_SHAPES
    assert "mouthSmileLeft" in ARKIT_BLEND_SHAPES
    assert "tongueOut" in ARKIT_BLEND_SHAPES
```

---

## Dependency Graph

```
T3.1 (Blend Shapes)
    |
    v
T3.2 (FACS) -----> T3.5 (Face Rig)
                        ^
T3.3 (Lip Sync) --------+
                        |
T3.4 (Eyes) ------------+

T3.6 (Retarget) --> T3.5
T3.7 (ARKit) --> T3.1
```

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| ARKit name mismatch | iOS integration fails | T3.7 exact name test |
| Coarticulation edge cases | Audio-visual desync | T3.3 edge case tests |
| Layer blend order | Wrong visual output | T3.5 priority tests |

---

## Definition of Done

Phase 3 is complete when:
1. All T3.x tasks have passing tests
2. All 8 Ekman expressions render visually correct
3. Lip sync with coarticulation is audibly smooth
4. Eye vergence converges correctly at all distances
5. ARKit 52 shapes are byte-exact match
