# Phase X2: XR Input Tracking — Architecture

**Tasks:** T-XR-2.1 through T-XR-2.7 (7 tasks)
**Effort:** 21-28 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X2 implements XR input tracking: HMD pose with prediction, controller input, action bindings, haptics, hand tracking (26 joints), and eye tracking.

---

## 2. HMD Tracking (`input/hmd.py`)

### HMDPose Component
```python
class HMDPose:
    # Predicted → Tracked → Atomic chain
    position: PredictedDescriptor[Vec3]
    orientation: PredictedDescriptor[Quat]
    
    # Velocity for prediction
    linear_velocity: TrackedDescriptor[Vec3]
    angular_velocity: TrackedDescriptor[Vec3]
    
    # Per-eye view matrices
    left_view: ComputedDescriptor[Mat4]
    right_view: ComputedDescriptor[Mat4]
    
    # Tracking state
    tracking_state: StateMeta  # unknown/tracking/limited/lost/disabled
    confidence: ExpiringDescriptor[float]  # 0.5s TTL
```

---

## 3. Controller Input (`input/controller.py`)

### XRController Component
```python
class XRController:
    hand: ImmutableDescriptor[Hand]  # LEFT, RIGHT
    
    # Poses
    grip_pose: PredictedDescriptor[Pose]
    aim_pose: PredictedDescriptor[Pose]
    
    # Analog inputs (Tracked → Validated → Range)
    trigger: RangeDescriptor[float]     # 0-1
    grip: RangeDescriptor[float]        # 0-1
    thumbstick_x: RangeDescriptor[float]  # -1 to 1
    thumbstick_y: RangeDescriptor[float]  # -1 to 1
    
    # Digital inputs
    button_a: TrackedDescriptor[bool]
    button_b: TrackedDescriptor[bool]
    thumbstick_click: TrackedDescriptor[bool]
    
    # Touch sensing
    trigger_touch: TrackedDescriptor[bool]
    thumbstick_touch: TrackedDescriptor[bool]
    
    # Haptic output (write-only)
    haptic: TransientDescriptor[HapticPulse]
```

---

## 4. Input Bindings (`input/bindings.py`)

### XR-Specific Actions
| Action | Default Binding |
|--------|-----------------|
| xr_grab | Grip button |
| xr_trigger | Trigger analog |
| xr_move | Thumbstick Y |
| xr_turn | Thumbstick X |
| xr_teleport | A button |
| xr_menu | Menu button |

### Decorators
- `@input_action` for digital inputs
- `@input_axis` for analog inputs

---

## 5. Haptics (`input/haptics.py`)

### HapticPulse
```python
class HapticPulse:
    amplitude: float  # 0-1
    duration: float   # seconds
    frequency: float  # Hz
```

Modes: pulse (single), continuous (sustained vibration).

---

## 6. Hand Tracking (`input/hand_tracking.py`)

### 26-Joint Model
```
Wrist
├── Thumb (metacarpal, proximal, distal, tip)
├── Index (metacarpal, proximal, intermediate, distal, tip)
├── Middle (...)
├── Ring (...)
└── Pinky (...)
```

### HandTracking Component
```python
class HandTracking:
    joints: BatchedDescriptor[JointArray]  # 26 joints
    joint_radii: TrackedDescriptor[float[26]]
    
    # Gestures
    is_pinching: TrackedDescriptor[bool]
    pinch_strength: RangeDescriptor[float]  # 0-1
    gesture: TrackedDescriptor[Gesture]  # open/fist/point/pinch
```

Update rate: >30Hz with interpolation.

---

## 7. Eye Tracking (`input/eye_tracking.py`)

### EyeTrackingData Component
```python
class EyeTrackingData:
    gaze_origin: TrackedDescriptor[Vec3]
    gaze_direction: TrackedDescriptor[Vec3]
    
    # Per-eye
    left_pupil_position: TrackedDescriptor[Vec2]
    right_pupil_position: TrackedDescriptor[Vec2]
    left_openness: RangeDescriptor[float]  # 0-1
    right_openness: RangeDescriptor[float]
    
    # Fixation
    fixation_point: TrackedDescriptor[Vec3]
    fixation_duration: TrackedDescriptor[float]
    
    # Calibration state
    calibration_state: StateMeta
```

---

## 8. Decorators

| Decorator | Purpose |
|-----------|---------|
| `@xr_tracked` | tracking_type, tracking_space |
| `@xr_controller` | hand, controller_type |
| `@xr_hand` | gesture_recognition |

---

## 9. Dependencies

- Phase X1: XR Runtime (OpenXR backend)
- Trinity: PredictedDescriptor, InterpolatedDescriptor, AtomicDescriptor, BatchedDescriptor, ExpiringDescriptor
