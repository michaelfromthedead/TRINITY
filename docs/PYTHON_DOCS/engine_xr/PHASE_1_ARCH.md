# PHASE 1 ARCHITECTURE: Core Runtime and Platform Foundation

## Phase Overview

Phase 1 establishes the foundational runtime abstraction and platform integration layer. This phase must complete before any other XR work because all other subsystems depend on runtime initialization, capability detection, and platform-specific feature access.

## Architectural Decisions

### ADR-XR-001: Runtime Abstraction Strategy

**Context**: The XR subsystem must support OpenXR, WebXR, and platform-specific runtimes while providing a unified API to higher layers.

**Decision**: Implement a three-tier runtime architecture:
1. **Abstract Base Class**: `XRRuntime` defines the unified interface
2. **Platform Implementations**: `OpenXRRuntime`, `WebXRRuntime`, platform-specific runtimes
3. **Mock Runtime**: `_MockRuntime` enables headset-free development

**Consequences**:
- Higher layers never interact with platform-specific APIs directly
- Runtime can be swapped at initialization time
- Testing proceeds without hardware

### ADR-XR-002: Session Lifecycle State Machine

**Context**: XR sessions have complex lifecycle requirements (initialization, running, paused, stopping, error recovery).

**Decision**: Implement explicit state machine with:
- **States**: IDLE, READY, RUNNING, PAUSED, STOPPING, STOPPED, ERROR
- **Transition Table**: `_VALID_TRANSITIONS` dict enforces legal state changes
- **Hooks**: `on_state_change`, `on_enter_*`, `on_exit_*` callbacks

**Consequences**:
- Invalid state transitions fail fast with clear errors
- Higher layers can register lifecycle callbacks
- Error recovery has defined paths

### ADR-XR-003: Capability Detection Model

**Context**: XR devices vary widely in capabilities (tracking, display, input, spatial features).

**Decision**: Use capability flags (`XRDeviceCapabilities`) with 26 feature types:
- **Tracking**: 6DOF head/controllers, hand tracking, eye tracking, face tracking, body tracking
- **Display**: Resolution, refresh rates, FOV, HDR, local dimming
- **Rendering**: Foveated rendering, multiview, space warp
- **Spatial**: Passthrough, depth sensing, plane detection, mesh detection, anchors, scene understanding
- **Input**: Controller type, haptics, finger tracking

**Consequences**:
- Feature gates prevent runtime errors on unsupported hardware
- Graceful degradation paths are explicit
- Capability database enables device-specific optimization

### ADR-XR-004: Platform Detection Priority

**Context**: Multiple XR runtimes may be available on a single system (e.g., SteamVR and Oculus on PC).

**Decision**: Implement priority-ordered detection:
1. Quest (standalone has no alternatives)
2. Vision Pro (visionOS is exclusive)
3. PSVR2 (PlayStation is exclusive)
4. SteamVR (user preference on PC)
5. OpenXR (universal fallback)

**Consequences**:
- Platform-specific features available when possible
- Consistent behavior per-platform
- User can override via configuration

### ADR-XR-005: Guardian/Boundary Abstraction

**Context**: Each platform has different boundary APIs (Chaperone, Guardian, Reference Space Bounds).

**Decision**: Implement abstract `GuardianSystem` with:
- **Math Layer**: Platform-agnostic geometry (shoelace area, point-in-polygon, proximity)
- **SDK Layer**: Platform-specific `request_bounds()` stub methods
- **Event Layer**: Proximity warnings via callbacks

**Consequences**:
- Boundary math is testable without hardware
- SDK integration adds data source only
- Uniform safety behavior across platforms

## Component Specifications

### XRRuntime (Abstract Base Class)

```
XRRuntime
├── initialize() -> bool
├── shutdown()
├── poll_events() -> List[XREvent]
├── wait_frame() -> bool
├── begin_frame() -> FrameState
├── end_frame()
├── get_head_pose(predicted_time) -> Pose
├── get_controller_pose(hand, predicted_time) -> Pose
├── get_view_info(eye) -> ViewInfo
├── submit_layers(layers: List[CompositorLayer])
└── Properties: is_running, display_refresh_rate, supported_features
```

### XRSession State Machine

```
States: IDLE -> READY -> RUNNING <-> PAUSED -> STOPPING -> STOPPED
                    \-> ERROR
                    
Valid Transitions:
  IDLE -> {READY, ERROR}
  READY -> {RUNNING, STOPPING, ERROR}
  RUNNING -> {PAUSED, STOPPING, ERROR}
  PAUSED -> {RUNNING, STOPPING, ERROR}
  STOPPING -> {STOPPED, ERROR}
  STOPPED -> {IDLE}
  ERROR -> {IDLE}
```

### XRDeviceCapabilities Structure

```
XRDeviceCapabilities
├── Tracking
│   ├── has_6dof_head: bool
│   ├── has_6dof_controllers: bool
│   ├── has_hand_tracking: bool
│   ├── has_eye_tracking: bool
│   ├── has_face_tracking: bool
│   └── has_body_tracking: bool
├── Display
│   ├── resolution: Tuple[int, int]
│   ├── supported_refresh_rates: List[float]
│   ├── field_of_view: Tuple[float, float, float, float]
│   ├── has_hdr: bool
│   └── has_local_dimming: bool
├── Rendering
│   ├── has_foveated_rendering: bool
│   ├── has_dynamic_foveation: bool
│   ├── has_multiview: bool
│   └── has_space_warp: bool
├── Spatial
│   ├── has_passthrough: bool
│   ├── has_color_passthrough: bool
│   ├── has_depth_sensing: bool
│   ├── has_plane_detection: bool
│   ├── has_mesh_detection: bool
│   ├── has_spatial_anchors: bool
│   ├── has_cloud_anchors: bool
│   └── has_scene_understanding: bool
├── Input
│   ├── controller_type: XRControllerType
│   ├── haptic_channels: int
│   └── has_finger_tracking: bool
└── Power
    ├── is_tethered: bool
    └── battery_capacity_wh: float
```

### GuardianSystem Architecture

```
GuardianSystem (Abstract)
├── request_bounds() -> PlayAreaBounds  [STUB - needs SDK]
├── set_custom_bounds(bounds) -> bool   [STUB - needs SDK]
├── recenter() -> bool                  [STUB - needs SDK]
├── _calculate_proximity(position) -> ProximityInfo  [REAL]
├── _point_to_segment_distance()                     [REAL]
├── _nearest_point_on_segment()                      [REAL]
├── update(head_position) -> ProximityLevel         [REAL]
├── get_passthrough_blend() -> float                [REAL]
└── get_warning_intensity() -> float                [REAL]

Implementations:
├── OpenXRGuardian
├── SteamVRGuardian
└── QuestGuardian
```

## Integration Points

### Dependencies (Incoming)
- `engine.core.math`: Vec3, Quat for pose representation
- `engine.xr.config`: XR_CONFIG for default values

### Dependents (Outgoing)
- All other XR subsystems depend on runtime initialization
- Rendering requires view info and frame timing
- Input requires controller/hand poses
- Spatial requires reference space

## Migration Path

### From Simulation to Real SDK

1. **OpenXR**: Replace simulation code with `pyopenxr` or ctypes bindings
   - `xrCreateInstance` -> `initialize()`
   - `xrWaitFrame` -> `wait_frame()`
   - `xrLocateSpace` -> `get_head_pose()`

2. **SteamVR**: Use `openvr` Python package
   - `IVRSystem::GetDeviceToAbsoluteTrackingPose` -> poses
   - `IVRChaperone::GetPlayAreaRect` -> guardian bounds

3. **Meta Quest**: Requires Android NDK bridge for OVR SDK
4. **visionOS**: Requires Swift interop layer
5. **PSVR2**: Requires PlayStation Partners SDK

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| pyopenxr bindings incomplete | Medium | High | Fallback to ctypes, contribute upstream |
| Platform-specific bugs | High | Medium | Per-platform test matrices |
| Session state corruption | Low | High | State machine enforces valid transitions |
| Guardian false positives | Medium | Medium | Configurable proximity thresholds |
