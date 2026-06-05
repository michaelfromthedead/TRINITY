# PHASE 2 ARCHITECTURE: Input System and Tracking

## Phase Overview

Phase 2 builds the input foundation that all interactive XR experiences require. This phase implements HMD tracking, controller input, hand tracking, eye tracking, and the action binding system. These components must be production-quality because they are on the critical path for every user interaction.

## Architectural Decisions

### ADR-XR-010: Tracking Data Flow

**Context**: Tracking data (HMD, controllers, hands, eyes) must flow from hardware to application with minimal latency and consistent timing.

**Decision**: Implement a pull-based tracking model with prediction:
1. **Poll Phase**: `poll_events()` collects raw tracking data from runtime
2. **Prediction Phase**: Pose prediction extrapolates to display time
3. **Access Phase**: Components query predicted poses via `get_*_pose()` methods

**Consequences**:
- Single poll per frame, not per-component
- Prediction compensates for rendering latency
- Tracking data is frame-consistent (all poses from same timestamp)

### ADR-XR-011: Controller State Model

**Context**: Controllers have analog inputs (trigger, grip, thumbstick) and digital inputs (buttons) with frame-relative states (pressed this frame, held, released this frame).

**Decision**: Implement three-state tracking per input:
- **Raw Value**: Analog value from hardware (0.0-1.0)
- **Processed Value**: After deadzone and curve application
- **Edge State**: Pressed/released detection via frame differencing

**Consequences**:
- Deadzone prevents drift without losing precision
- Edge detection enables button-press events
- Analog values remain available for proportional control

### ADR-XR-012: Hand Tracking Joint Model

**Context**: Hand tracking provides 26 joints per hand following the OpenXR standard.

**Decision**: Use OpenXR joint enumeration directly:
```
PALM, WRIST
THUMB: METACARPAL, PROXIMAL, DISTAL, TIP
INDEX/MIDDLE/RING/LITTLE: METACARPAL, PROXIMAL, INTERMEDIATE, DISTAL, TIP
```

**Consequences**:
- Direct mapping to OpenXR hand tracking extension
- Consistent with Meta Quest and other tracking systems
- Joint indices stable across updates

### ADR-XR-013: Gesture Recognition Pipeline

**Context**: Hand gestures must be recognized reliably with configurable thresholds.

**Decision**: Implement a three-stage pipeline:
1. **Feature Extraction**: Finger curl, spread, tip distances
2. **Gesture Matching**: Pattern matching against gesture definitions
3. **Smoothing**: History buffer prevents flicker

**Consequences**:
- Custom gestures can be registered at runtime
- Confidence values enable thresholded activation
- History smoothing adds 2-3 frame latency

### ADR-XR-014: Eye Tracking Algorithm Selection

**Context**: Eye tracking data requires processing for fixation/saccade detection.

**Decision**: Use I-VT (Velocity-Threshold Identification):
- Gaze velocity below threshold = fixation
- Gaze velocity above threshold = saccade
- Blink detected via eye openness drop

**Consequences**:
- Simple, well-understood algorithm
- Configurable velocity threshold
- Works with varying eye tracker sample rates

### ADR-XR-015: Action Binding Architecture

**Context**: Input should be abstract (game actions) not concrete (button A on controller X).

**Decision**: Implement OpenXR-style action system:
1. **Actions**: Named, typed game inputs (`grab`, `teleport`, `menu`)
2. **Bindings**: Map hardware inputs to actions with threshold/scale
3. **Profiles**: Per-controller-type binding sets

**Consequences**:
- Same code works across controller types
- User rebinding possible at runtime
- Decorators (`@xr_action`, `@xr_axis`) simplify binding

### ADR-XR-016: Haptic Feedback Model

**Context**: Haptic feedback must support simple rumble, waveforms, and patterns.

**Decision**: Implement three haptic tiers:
1. **Simple**: Amplitude + duration + frequency
2. **Waveform**: Sine, square, triangle, click, buzz shapes
3. **Pattern**: Sequenced effects with timing and looping

**Consequences**:
- Simple API covers 90% of use cases
- Waveforms enable nuanced feedback
- Patterns enable complex notifications

## Component Specifications

### HMD Tracking Component

```
HMD
├── State: INITIALIZING, TRACKING, LIMITED, LOST, DISABLED
├── update_pose(position, orientation, linear_vel, angular_vel)
├── get_predicted_pose(time_offset) -> Pose
├── get_view_matrix(eye, ipd_offset) -> Mat4
├── Properties
│   ├── position: Vec3
│   ├── orientation: Quat
│   ├── linear_velocity: Vec3
│   ├── angular_velocity: Vec3
│   └── tracking_state: HMDTrackingState
└── Callbacks
    └── on_tracking_state_changed(old_state, new_state)
```

### Controller Component

```
XRController
├── update(platform_state)
├── Button State
│   ├── is_button_down(button) -> bool
│   ├── is_button_pressed(button) -> bool  # This frame
│   ├── is_button_released(button) -> bool # This frame
│   └── is_button_touched(button) -> bool  # Capacitive
├── Analog State
│   ├── get_trigger() -> float (0-1)
│   ├── get_grip() -> float (0-1)
│   ├── get_thumbstick() -> Vec2 (-1 to 1)
│   └── get_thumbstick_with_deadzone() -> Vec2
├── Pose State
│   ├── grip_pose: RigidTransform
│   └── aim_pose: RigidTransform
├── Haptics
│   └── play_haptic(effect: HapticEffect)
└── Properties
    ├── hand: XRHand (LEFT, RIGHT)
    ├── controller_type: XRControllerType
    └── capabilities: ControllerCapabilities
```

### Hand Tracking Component

```
HandTrackingData
├── update(joint_data: List[JointData])
├── Joint Access
│   ├── get_joint(joint: HandJoint) -> JointData
│   ├── get_finger_curl(finger: FingerName) -> float (0-1)
│   ├── get_finger_spread(finger: FingerName) -> float (-1 to 1)
│   └── get_palm_pose() -> RigidTransform
├── Gesture State
│   ├── get_pinch_strength() -> float (0-1)
│   ├── get_grip_strength() -> float (0-1)
│   └── is_gesture_active(gesture: GestureType) -> bool
├── Properties
│   ├── hand: XRHand
│   ├── tracking_confidence: float
│   └── joint_count: int (26)
└── Serialization
    ├── to_dict() -> dict
    └── from_dict(data: dict) -> HandTrackingData

GestureRecognizer
├── register_gesture(name, detector_fn)
├── recognize(hand_data) -> List[GestureResult]
└── Built-in Gestures
    ├── PINCH: thumb-index distance < 5cm
    ├── POINT: index extended, others curled
    ├── FIST: all fingers curled > 0.8
    ├── OPEN_HAND: all fingers extended < 0.2
    └── THUMBS_UP: thumb up, others curled
```

### Eye Tracking Component

```
EyeTrackingData
├── update(left_eye, right_eye)
├── Eye State
│   ├── get_gaze_ray(eye: EyeId) -> Ray
│   ├── get_gaze_point(depth: float) -> Vec3
│   ├── get_eye_openness(eye: EyeId) -> float
│   ├── get_pupil_diameter(eye: EyeId) -> float
│   └── get_vergence_distance() -> float
├── Detection State
│   ├── gaze_state: GazeState (FIXATION, SACCADE, BLINK)
│   ├── fixation: FixationData (position, duration)
│   └── blink: BlinkData (start_time, duration)
└── Calibration
    ├── calibration_state: CalibrationState
    ├── start_calibration(point_count: int)
    ├── record_calibration_point(point: Vec3)
    └── get_calibration_error() -> float

FixationDetector
├── velocity_threshold: float (default 30 deg/s)
├── update(gaze_direction, delta_time) -> GazeState
└── get_fixation() -> FixationData

BlinkDetector
├── openness_threshold: float (default 0.3)
├── update(eye_openness) -> bool
└── get_last_blink() -> BlinkData
```

### Action Binding System

```
XRActionRegistry (Singleton)
├── register_action(name, action_type) -> XRAction
├── bind_action(action_name, source, binding) -> bool
├── get_action_value(action_name) -> ActionValue
├── apply_profile(profile: XRInputProfile)
└── Decorators
    ├── @xr_action(action_name): Bind method to action
    └── @xr_axis(action_name, axis): Bind method to axis

XRAction
├── name: str
├── type: XRActionType (BOOLEAN, FLOAT, VECTOR2, POSE, HAPTIC)
├── get_value() -> T
├── is_active() -> bool
└── bindings: List[XRActionBinding]

XRInputProfile
├── name: str (e.g., "valve_index_controller")
├── bindings: Dict[str, List[XRActionBinding]]
└── apply() -> None
```

### Haptic System

```
HapticManager
├── play(hand, effect: HapticEffect)
├── play_pattern(hand, pattern: HapticPattern)
├── stop(hand)
├── set_global_amplitude(scale: float)
└── get_capabilities(hand) -> HapticCapabilities

HapticEffect
├── amplitude: float (0-1)
├── duration_ms: int
├── frequency: float (Hz)
├── waveform: HapticWaveform
├── fade_in_ms: int
└── fade_out_ms: int

HapticPattern
├── name: str
├── effects: List[Tuple[HapticEffect, delay_ms]]
├── loop_count: int
└── Built-in Patterns
    ├── HEARTBEAT: Two pulses with gap
    ├── SUCCESS: Rising pulse sequence
    ├── ERROR: Buzz pattern
    └── NOTIFICATION: Double tap
```

## Integration Points

### Dependencies (Incoming)
- Phase 1: Runtime initialization, session state, capability detection
- `engine.core.math`: Vec2, Vec3, Quat, Transform

### Dependents (Outgoing)
- Phase 3: Avatars consume hand tracking for animation
- Phase 4: Locomotion consumes controller input
- Phase 5: Spatial UI consumes ray/poke/gaze input

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          XR Runtime                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │   HMD    │ │Controller│ │   Hand   │ │   Eye    │           │
│  │ Tracking │ │ Tracking │ │ Tracking │ │ Tracking │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌──────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐
│   HMD    │ │ Controller │ │  Hand     │ │   Eye     │
│Component │ │ Component  │ │ Component │ │ Component │
└────┬─────┘ └─────┬──────┘ └─────┬─────┘ └─────┬─────┘
     │             │              │             │
     │             ▼              ▼             │
     │     ┌──────────────┐ ┌──────────┐       │
     │     │   Action     │ │ Gesture  │       │
     │     │   Binding    │ │Recognizer│       │
     │     └──────┬───────┘ └────┬─────┘       │
     │            │              │             │
     └────────────┴──────────────┴─────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Application Code  │
              │   (via decorators)  │
              └─────────────────────┘
```

## Performance Requirements

| Component | Update Rate | Latency Budget |
|-----------|-------------|----------------|
| HMD Tracking | 90-120 Hz | <11ms |
| Controller Tracking | 90-120 Hz | <11ms |
| Hand Tracking | 30-60 Hz | <33ms |
| Eye Tracking | 90-120 Hz | <11ms |
| Gesture Recognition | 30-60 Hz | <16ms |
| Action Binding | 90-120 Hz | <1ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hand tracking jitter | High | Medium | Smoothing filter, confidence thresholds |
| Eye tracking calibration drift | Medium | Medium | Periodic recalibration prompts |
| Controller deadzone mismatch | Medium | Low | Per-controller deadzone configuration |
| Action binding conflicts | Low | Medium | Binding validation at registration |
