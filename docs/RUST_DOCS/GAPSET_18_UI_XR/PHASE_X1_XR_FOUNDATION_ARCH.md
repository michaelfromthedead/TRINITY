# Phase X1: XR Runtime Foundation — Architecture

**Tasks:** T-XR-1.1 through T-XR-1.4 (4 tasks)
**Effort:** 12-17 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X1 establishes the XR runtime abstraction layer, OpenXR backend, device capability detection, and session state machine.

---

## 2. Runtime Abstraction (`runtime/xr_runtime.py`)

### XRRuntimeState Resource
```python
class XRRuntimeState:
    # Immutable runtime info
    runtime_name: ImmutableDescriptor[str]
    runtime_version: ImmutableDescriptor[str]
    
    # Session state (observable for UI binding)
    session_state: TrackedDescriptor[SessionState]  # + ObservableDescriptor
    
    # Device capabilities
    capabilities: ImmutableDescriptor[XRCapabilities]
    
    # Display specs
    display_width: ImmutableDescriptor[int]
    display_height: ImmutableDescriptor[int]
    refresh_rate: ImmutableDescriptor[float]
```

### Backend Interface
```python
class XRBackend(ABC):
    def create_instance(self) -> XRInstance
    def select_system(self) -> XRSystem
    def create_session(self) -> XRSession
    def poll_events(self) -> list[XREvent]
    def get_poses(self, predicted_time: float) -> XRPoses
```

---

## 3. OpenXR Backend (`runtime/openxr.py`)

### OpenXR 1.0+ Compliance
- Instance creation with required extensions
- System selection (HMD device)
- Session lifecycle: idle → ready → running → stopping
- Reference spaces: VIEW, LOCAL, STAGE

### Pose Polling
- HMD pose at 250Hz+ for smooth rendering
- Controller poses synchronized with HMD
- Predicted time for motion-to-photon optimization

---

## 4. Capability Detection (`runtime/capabilities.py`)

| Capability | Query Source |
|------------|--------------|
| supports_hand_tracking | XR_EXT_hand_tracking extension |
| supports_eye_tracking | XR_EXT_eye_gaze_interaction |
| supports_passthrough | XR_FB_passthrough |
| supports_spatial_mesh | XR_MSFT_scene_understanding |
| max_controllers | System properties |
| display_refresh_rate | XR_FB_display_refresh_rate |
| field_of_view | View configuration |

Capabilities cached as `ImmutableDescriptor` after init.

---

## 5. Session State Machine (`runtime/session.py`)

### States
```
IDLE → READY → RUNNING → STOPPING → IDLE
```

### StateMeta Configuration
```python
@state_machine(transitions={
    "IDLE": ["READY"],
    "READY": ["RUNNING", "IDLE"],
    "RUNNING": ["STOPPING"],
    "STOPPING": ["IDLE"],
})
class XRSessionState: ...
```

### Hooks
- `@on_enter(RUNNING)`: Begin frame loop
- `@on_exit(RUNNING)`: Pause rendering
- State changes fire `ObservableDescriptor` to UI/game

---

## 6. Dependencies

- Foundation: Registry, Tracker, EventLog, Mirror
- Trinity: ResourceMeta, SystemMeta, ComponentMeta
- External: OpenXR SDK (libOpenXR_loader)
