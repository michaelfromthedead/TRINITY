# Phase X6: XR Avatars — Architecture

**Tasks:** T-XR-6.1 through T-XR-6.6 (6 tasks)
**Effort:** 17-24 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X6 implements full-body avatars for XR: IK-driven body, FABRIK solver, hand/finger animation, face tracking, and calibration.

---

## 2. Avatar Component (`avatars/avatar.py`)

### XRAvatar Component
```python
class XRAvatar:
    # IK targets (from HMD and controllers)
    head_target: TrackedDescriptor[Pose]
    left_hand_target: TrackedDescriptor[Pose]
    right_hand_target: TrackedDescriptor[Pose]
    
    # Computed body
    torso_pose: ComputedDescriptor[Pose]  # Estimated from head
    
    # Visibility
    visible_to_self: TrackedDescriptor[bool]
    visible_to_others: TrackedDescriptor[bool]
    
    # Calibration
    height: ImmutableDescriptor[float]
    arm_span: ImmutableDescriptor[float]
```

Update rate: 90Hz for body tracking.

---

## 3. IK Solver (`avatars/ik_solver.py`)

### FABRIK Algorithm
Forward And Backward Reaching Inverse Kinematics.

```
1. Forward pass: Move joints toward target
2. Backward pass: Move joints back toward root
3. Repeat until converged (3-5 iterations)
```

### Chain Definitions
| Chain | Joints |
|-------|--------|
| Left Arm | shoulder → elbow → wrist |
| Right Arm | shoulder → elbow → wrist |
| Spine | hip → spine → neck → head |
| Left Leg | hip → knee → ankle |
| Right Leg | hip → knee → ankle |

### Constraints
Joint angle limits prevent unnatural poses.

### CCD Fallback
Cyclic Coordinate Descent for fingers (simpler, faster).

### Performance
Solve time <0.1ms per chain at 90Hz.

---

## 4. Hand Animation (`avatars/hand_animator.py`)

### Finger Curl Values
```python
finger_curls: dict[Finger, float]  # 0 = open, 1 = closed
# thumb, index, middle, ring, pinky
```

### Input Sources
| Source | Used For |
|--------|----------|
| Controller grip/trigger | Analog curl estimate |
| Hand tracking joints | Full finger pose |

### Blend Tree
Posture blending: open ↔ point ↔ fist ↔ pinch.

### Display Modes
- Hand (rendered hand mesh)
- Controller (render controller model)
- Tool (render held object at hand)

---

## 5. Face Tracking (`avatars/face_tracking.py`)

### FACS-Compatible Blend Shapes
| Blend Shape | Description |
|-------------|-------------|
| jawOpen | Mouth open |
| browRaiseL/R | Eyebrow raise |
| smile | Corners of mouth up |
| squint | Eyes narrowed |
| ... | (52 standard shapes) |

### Input
Eye tracking cameras provide face data.

### Update Rate
30Hz (sufficient for expression).

---

## 6. Calibration (`avatars/calibration.py`)

### T-Pose / A-Pose Calibration
At session start, user holds standard pose.

### Measurements
- Height
- Arm span
- Shoulder width (derived)

### Application
Scales IK skeleton proportions to user.

### Persistence
Calibration data saved via Foundation Serializer.

### Accuracy
Within 1cm of actual measurements.

---

## 7. Decorators

| Decorator | Configuration |
|-----------|---------------|
| `@xr_avatar` | ik_enabled, network_sync |
| `@xr_ik_target` | target_type, bone_chain |

---

## 8. Dependencies

- Phase X1: XR Runtime
- Phase X2: HMD pose, controller input, hand tracking
- Phase X4: Interaction for hand pose
- S15: Rust math library (vector/quaternion ops)
