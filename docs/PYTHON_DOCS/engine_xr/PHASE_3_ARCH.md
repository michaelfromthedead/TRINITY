# PHASE 3 ARCHITECTURE: Avatar System

## Phase Overview

Phase 3 implements the avatar representation layer that bridges physical user body to virtual presence. This phase covers inverse kinematics, hand animation, face tracking, and body calibration. The avatar system is the primary vehicle for social presence in multiplayer XR.

## Architectural Decisions

### ADR-XR-020: Multi-Solver IK Strategy

**Context**: Different body parts have different IK requirements (arm vs spine vs procedural creatures).

**Decision**: Implement three solver algorithms with factory selection:
1. **TwoBone**: Analytical solution for arms and legs (O(1), exact)
2. **FABRIK**: Forward And Backward Reaching for chains (O(n) per iteration)
3. **CCD**: Cyclic Coordinate Descent for constrained joints

**Consequences**:
- Optimal solver per body part
- Factory pattern hides algorithm selection
- Custom solvers can be added for procedural animation

### ADR-XR-021: ARKit Blend Shape Standard

**Context**: Face tracking must interoperate with existing avatar tools and assets.

**Decision**: Use ARKit's 52 blend shape standard as the canonical face representation:
- Eyes: lookUp/Down/Left/Right, blink, widen, squint
- Brows: down, inner/outer up
- Mouth: 26 shapes covering jaw, lips, tongue
- Nose/Cheek: sneer, puff, funnel

**Consequences**:
- Compatible with Unreal, Unity, Blender rigs
- Future Meta/Apple face tracking maps directly
- Artists can use standard blend shape workflows

### ADR-XR-022: Hand Pose Library Pattern

**Context**: Hand poses need both predefined options and custom registration.

**Decision**: Implement static pose library with runtime extension:
- **Predefined**: OPEN, FIST, POINT, PINCH, GRIP, THUMBS_UP, PEACE, ROCK, OK
- **Custom**: `PoseLibrary.register(name, HandPose)` for app-specific poses
- **Blending**: Linear interpolation between poses

**Consequences**:
- Common poses work out of box
- Game-specific gestures easily added
- Smooth transitions via interpolation

### ADR-XR-023: Body Estimation from Three Points

**Context**: VR provides only HMD and two controllers, but avatars need full body.

**Decision**: Estimate body from three tracked points:
- **Pelvis**: 50% of calibrated height, follows head yaw only
- **Chest**: Interpolated between head and pelvis
- **Feet**: Procedural placement with configurable stride width

**Consequences**:
- No additional trackers required
- Body estimation is approximate but sufficient for most use cases
- Full body tracking can override estimation when available

### ADR-XR-024: Calibration State Machine

**Context**: Calibration must guide users through measurement steps and handle failures.

**Decision**: Implement explicit calibration state machine:
- **States**: NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED
- **Steps**: FLOOR_DETECTION, HEIGHT_MEASUREMENT, ARM_SPAN_MEASUREMENT, T_POSE, A_POSE
- **Persistence**: Save/load calibration data with version checking

**Consequences**:
- Clear user guidance through calibration
- Partial calibration can succeed (floor but not arm span)
- Calibration persists across sessions

### ADR-XR-025: Personal Space Enforcement

**Context**: Multiplayer XR needs physical boundary between users.

**Decision**: Implement personal space with three responses:
1. **Push**: Generate push vector when invaded
2. **Fade**: Reduce invading avatar opacity
3. **Indicator**: Show visual boundary ring

**Consequences**:
- Users cannot clip through each other
- Gradual response prevents jarring teleport
- Configurable per-user comfort settings

### ADR-XR-026: Network State Optimization

**Context**: Avatar state must sync in multiplayer with limited bandwidth.

**Decision**: Implement bandwidth-aware serialization:
- Only non-zero blend shapes sent (most faces use <20 shapes)
- Quantized positions/rotations where precision not critical
- Separate update rates for different body parts

**Consequences**:
- Face sync: 60-80 bytes/frame instead of 400+
- Full body: <200 bytes/frame
- Configurable precision vs bandwidth tradeoff

## Component Specifications

### IK Solver System

```
IKSolver (Abstract Base)
├── solve(chain: IKChain, target: IKTarget) -> List[Transform]
└── Properties
    ├── max_iterations: int
    └── tolerance: float

TwoBoneSolver(IKSolver)
├── solve() - Law of cosines analytical solution
├── Support for pole target (elbow/knee direction)
└── Rotation-from-matrix to quaternion conversion

FABRIKSolver(IKSolver)
├── solve() - Forward/backward reaching iterative
├── Handle unreachable targets by stretching
└── O(n) per iteration, fast for long chains

CCDSolver(IKSolver)
├── solve() - Cyclic coordinate descent
├── Apply joint angle limits via clamp_rotation()
└── Good for constrained joints (knee, elbow)

IKChain
├── joints: List[IKJoint]
├── target: IKTarget
└── pole_target: Optional[Vec3]

IKJoint
├── position: Vec3
├── rotation: Quat
├── length: float
├── angle_limits: Tuple[float, float, float, float]
├── twist_axis: Vec3
└── swing_axis: Vec3

Factory Functions
├── create_arm_chain() -> IKChain
├── create_leg_chain() -> IKChain
└── create_solver(type: IKSolverType) -> IKSolver
```

### Avatar Component

```
XRAvatar
├── IK Targets (tracked from input)
│   ├── head: IKTarget
│   ├── left_hand: IKTarget
│   └── right_hand: IKTarget
├── Estimated Body Parts
│   ├── pelvis: Transform
│   ├── chest: Transform
│   ├── left_foot: Transform
│   └── right_foot: Transform
├── Calibration Data
│   ├── player_height: float
│   ├── arm_span: float
│   ├── floor_level: float
│   └── shoulder_width: float
├── Personal Space
│   ├── radius: float (default 0.5m)
│   ├── check_invasion(other_position) -> bool
│   ├── get_push_vector(other_position) -> Vec3
│   └── get_fade_alpha(distance) -> float
├── Display
│   ├── visibility: AvatarVisibility
│   ├── display_mode: DisplayMode
│   └── name_tag: Optional[str]
└── Network
    ├── get_network_state() -> dict
    └── apply_network_state(state: dict)

AvatarVisibility Enum
├── VISIBLE
├── HIDDEN
├── SELF_HIDDEN (others see, self doesn't)
└── OTHERS_HIDDEN (self sees, others don't)

DisplayMode Enum
├── CONTROLLER (show controller models)
├── HAND (show hand models)
└── TOOL (show held tool)
```

### Hand Animation Component

```
AvatarHand
├── Pose State
│   ├── current_pose: HandPose
│   ├── target_pose: HandPose
│   └── blend_speed: float
├── Input Sources
│   ├── set_pose_from_controller(trigger, grip) -> None
│   ├── set_pose_from_hand_tracking(joints: List[JointData]) -> None
│   └── set_pose_from_name(pose_name: str) -> None
├── Finger Access
│   ├── get_finger_curl(finger: FingerName) -> float
│   ├── get_finger_spread(finger: FingerName) -> float
│   └── get_finger_twist(finger: FingerName) -> float
├── Interaction
│   ├── get_grip_strength() -> float
│   └── get_pinch_strength() -> float
└── Network
    ├── get_network_state() -> dict
    └── apply_network_state(state: dict)

PoseLibrary (Static)
├── get_pose(name: str) -> HandPose
├── register_pose(name: str, pose: HandPose)
├── interpolate(a: HandPose, b: HandPose, t: float) -> HandPose
└── Predefined Poses
    ├── OPEN, FIST, POINT, PINCH, GRIP
    └── THUMBS_UP, PEACE, ROCK, OK

HandPose
├── thumb: FingerCurl
├── index: FingerCurl
├── middle: FingerCurl
├── ring: FingerCurl
├── pinky: FingerCurl
└── wrist: Optional[Transform]

FingerCurl
├── curl: float (0-1)
├── spread: float (-1 to 1)
└── twist: float (-1 to 1)
```

### Face Tracking Component

```
FaceTracking
├── Blend Shape State
│   ├── weights: Dict[BlendShapeType, float] (52 shapes)
│   ├── set_weight(shape: BlendShapeType, value: float)
│   ├── get_weight(shape: BlendShapeType) -> float
│   └── apply_expression(expression: ExpressionType)
├── Eye Integration
│   ├── update_from_eye_tracking(left: EyeData, right: EyeData)
│   ├── apply_gaze_blend_shapes(direction: Vec3)
│   └── apply_blink_blend_shapes(openness: float)
├── Lip Sync
│   ├── set_viseme(viseme: Viseme, strength: float)
│   ├── update_from_audio(audio_features: LipSyncData)
│   └── Visemes: AA, AH, AO, AW, CH, EE, EH, ER, IH, K, N, OH, OO, R, S, SH, T, TH, W, WW, SILENCE
├── Auto Animation
│   ├── enable_auto_blink: bool
│   ├── blink_interval_range: Tuple[float, float]
│   └── update_auto_blink(delta_time)
├── Expression Presets
│   ├── NEUTRAL, HAPPY, SAD, ANGRY
│   └── SURPRISED, DISGUSTED, SCARED, THINKING
└── Network
    ├── get_network_state() -> dict (only non-zero shapes)
    └── apply_network_state(state: dict)

BlendShapeController
├── target_weights: Dict[BlendShapeType, float]
├── current_weights: Dict[BlendShapeType, float]
├── blend_speed: float
└── update(delta_time) -> None
```

### Calibration System

```
AvatarCalibration
├── State Machine
│   ├── state: CalibrationState
│   ├── current_step: CalibrationStep
│   └── progress: float (0-1 per step)
├── Guided Calibration
│   ├── start_calibration() -> None
│   ├── advance_step() -> bool
│   ├── record_sample(hmd_pose, hand_poses) -> None
│   └── complete_calibration() -> CalibrationData
├── Quick Calibration
│   └── quick_calibrate(hmd_pose, hand_poses) -> CalibrationData
├── Manual Calibration
│   └── set_values(height, arm_span, floor) -> CalibrationData
├── Persistence
│   ├── save(path: str) -> bool
│   ├── load(path: str) -> CalibrationData
│   └── VERSION: int = 1
└── Progress Queries
    ├── get_current_instruction() -> str
    ├── get_step_progress(step: CalibrationStep) -> float
    └── get_overall_progress() -> float

CalibrationData
├── player_height: float
├── arm_span: float
├── floor_level: float
├── shoulder_width: float (derived)
├── leg_length: float (derived)
├── torso_length: float (derived)
└── Derived from human proportions:
    ├── eye_height = 0.94 * player_height
    ├── shoulder_width = 0.24 * arm_span
    ├── arm_length = 0.35 * arm_span
    ├── leg_length = 0.50 * player_height
    └── torso_length = 0.32 * player_height

CalibrationStep Enum
├── FLOOR_DETECTION (touch floor or average HMD - 1.6m)
├── HEIGHT_MEASUREMENT (HMD height / 0.94)
├── ARM_SPAN_MEASUREMENT (T-pose hand distance)
├── T_POSE (arms out, for reference)
└── A_POSE (arms down at angle, for animation default)
```

## Integration Points

### Dependencies (Incoming)
- Phase 2: Hand tracking joint data, eye tracking gaze data
- `engine.core.math`: Vec3, Quat, Transform

### Dependents (Outgoing)
- Renderer: Avatar meshes, blend shapes, bone transforms
- Network: Multiplayer avatar sync

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Input System (Phase 2)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │   HMD    │ │Controller│ │   Hand   │ │   Eye    │       │
│  │   Pose   │ │   Input  │ │  Joints  │ │   Gaze   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                       Avatar System                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Calibration │  │  IK Solver  │  │ Pose Library│          │
│  │   Data      │  │             │  │             │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                       XRAvatar                          ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │Head/Hands│  │  Body    │  │  Face    │              ││
│  │  │ (IK)     │  │(Estimated)│  │(BlendSh.)│              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
        │                │                │
        ▼                ▼                ▼
    Renderer         Network          Personal
    (Skinning)       (Sync)           Space
```

## Performance Requirements

| Component | Update Rate | CPU Budget |
|-----------|-------------|------------|
| IK Solving | 90 Hz | <0.5ms |
| Hand Animation | 90 Hz | <0.2ms |
| Face Tracking | 60 Hz | <0.3ms |
| Body Estimation | 90 Hz | <0.2ms |
| Network Sync | 30 Hz | <0.1ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| IK instability near limits | Medium | Medium | Joint limit clamping, solver iteration caps |
| Calibration user confusion | Medium | Low | Clear visual/audio instructions |
| Face tracking jitter | High | Medium | Blend shape smoothing |
| Network bandwidth spikes | Medium | Medium | Delta compression, rate limiting |
