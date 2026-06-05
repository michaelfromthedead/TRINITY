# PHASE 1 TODO: Core Runtime and Platform Foundation

## Overview

Phase 1 implements native SDK bindings for the runtime abstraction layer. The API design is complete; this phase adds real hardware communication.

## Tasks

### T-XR-1.1: OpenXR Native Bindings

**Priority**: Critical
**Effort**: Large (40 hours)
**Dependencies**: None

**Description**: Replace simulation code in `openxr.py` with real OpenXR SDK calls via `pyopenxr` or ctypes.

**Subtasks**:
- [ ] T-XR-1.1.1: Install and configure `pyopenxr` package
- [ ] T-XR-1.1.2: Implement `xrCreateInstance` in `initialize()`
- [ ] T-XR-1.1.3: Implement `xrGetSystem` for device enumeration
- [ ] T-XR-1.1.4: Implement `xrCreateSession` for session creation
- [ ] T-XR-1.1.5: Implement `xrWaitFrame` / `xrBeginFrame` / `xrEndFrame` cycle
- [ ] T-XR-1.1.6: Implement `xrLocateSpace` for head pose
- [ ] T-XR-1.1.7: Implement `xrLocateSpace` for controller poses
- [ ] T-XR-1.1.8: Implement `xrGetReferenceSpaceBoundsRect` for guardian

**Acceptance Criteria**:
- [ ] `initialize()` returns True on OpenXR-capable hardware
- [ ] `get_head_pose()` returns actual HMD position/orientation
- [ ] `get_controller_pose()` returns actual controller positions
- [ ] `wait_frame()` blocks until compositor ready (no `time.sleep`)
- [ ] Frame timing matches display refresh rate (90Hz = 11.11ms)

**Files**:
- `engine/xr/runtime/openxr.py`

---

### T-XR-1.2: SteamVR Integration

**Priority**: High
**Effort**: Medium (24 hours)
**Dependencies**: T-XR-1.1 (optional, can parallelize)

**Description**: Implement SteamVR runtime using `openvr` Python package.

**Subtasks**:
- [ ] T-XR-1.2.1: Install and configure `openvr` package
- [ ] T-XR-1.2.2: Implement `IVRSystem` initialization
- [ ] T-XR-1.2.3: Implement `GetDeviceToAbsoluteTrackingPose` for poses
- [ ] T-XR-1.2.4: Implement `IVRChaperone::GetPlayAreaRect` for guardian
- [ ] T-XR-1.2.5: Implement `IVRCompositor::Submit` for frame submission

**Acceptance Criteria**:
- [ ] SteamVR detected and initialized on Windows/Linux with SteamVR installed
- [ ] Tracking data matches SteamVR overlay diagnostics
- [ ] Chaperone bounds match SteamVR room setup

**Files**:
- `engine/xr/runtime/steamvr.py` (new file)
- `engine/xr/platform/platform_integration.py`

---

### T-XR-1.3: Session State Machine Hardening

**Priority**: High
**Effort**: Small (8 hours)
**Dependencies**: None

**Description**: Add error recovery and timeout handling to session state machine.

**Subtasks**:
- [ ] T-XR-1.3.1: Implement automatic retry for transient initialization failures
- [ ] T-XR-1.3.2: Add timeout detection for stuck states (READY without RUNNING)
- [ ] T-XR-1.3.3: Implement graceful degradation on partial initialization
- [ ] T-XR-1.3.4: Add metrics collection for state transition timing

**Acceptance Criteria**:
- [ ] Session recovers from transient USB disconnection
- [ ] Stuck state detected within 5 seconds and logged
- [ ] Partial initialization (e.g., no controllers) does not block session
- [ ] State transition times logged for performance analysis

**Files**:
- `engine/xr/runtime/session.py`

---

### T-XR-1.4: Capability Detection Validation

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1, T-XR-1.2

**Description**: Validate capability detection against real hardware and expand device database.

**Subtasks**:
- [ ] T-XR-1.4.1: Test capability reporting on Quest 3
- [ ] T-XR-1.4.2: Test capability reporting on Valve Index
- [ ] T-XR-1.4.3: Test capability reporting on HP Reverb G2
- [ ] T-XR-1.4.4: Add runtime capability probing vs static database
- [ ] T-XR-1.4.5: Document capability gaps per device

**Acceptance Criteria**:
- [ ] Capability flags match device spec sheets
- [ ] Runtime probing detects features not in static database
- [ ] Capability mismatches logged as warnings

**Files**:
- `engine/xr/runtime/capabilities.py`
- `engine/xr/platform/platform_integration.py`

---

### T-XR-1.5: Guardian SDK Integration

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1, T-XR-1.2

**Description**: Connect guardian math to real SDK boundary data.

**Subtasks**:
- [ ] T-XR-1.5.1: Implement OpenXR `xrGetReferenceSpaceBoundsRect` call
- [ ] T-XR-1.5.2: Implement SteamVR `IVRChaperone::GetPlayAreaRect` call
- [ ] T-XR-1.5.3: Implement Quest `OVR_Guardian.OvrBoundary_GetGeometry` (via OpenXR extension)
- [ ] T-XR-1.5.4: Test boundary polygon correctness against room setup
- [ ] T-XR-1.5.5: Validate proximity warnings trigger at correct distances

**Acceptance Criteria**:
- [ ] `request_bounds()` returns polygon matching room setup
- [ ] Proximity warnings match platform-native guardian
- [ ] Passthrough blend activates at boundary

**Files**:
- `engine/xr/platform/guardian.py`

---

### T-XR-1.6: Mock Runtime Enhancement

**Priority**: Low
**Effort**: Small (8 hours)
**Dependencies**: None

**Description**: Enhance mock runtime for automated testing and CI.

**Subtasks**:
- [ ] T-XR-1.6.1: Add scripted pose sequences for testing
- [ ] T-XR-1.6.2: Add frame timing simulation (dropped frames, stutters)
- [ ] T-XR-1.6.3: Add capability override for testing degraded modes
- [ ] T-XR-1.6.4: Add event injection for testing callbacks

**Acceptance Criteria**:
- [ ] Tests can simulate head movement sequences
- [ ] Tests can simulate dropped frames and verify reprojection
- [ ] Tests can simulate device with limited capabilities

**Files**:
- `engine/xr/runtime/xr_runtime.py` (_MockRuntime class)

---

### T-XR-1.7: Social Services Stub Completion

**Priority**: Low
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Complete social services stub interfaces to match SDK patterns.

**Subtasks**:
- [ ] T-XR-1.7.1: Document Meta Platform SDK method signatures
- [ ] T-XR-1.7.2: Document Steam Friends/Matchmaking method signatures
- [ ] T-XR-1.7.3: Add placeholder async patterns for network operations
- [ ] T-XR-1.7.4: Add error handling patterns for auth failures

**Acceptance Criteria**:
- [ ] Stub method signatures match SDK documentation
- [ ] Async patterns established for future implementation
- [ ] Error codes documented per platform

**Files**:
- `engine/xr/platform/social.py`

---

## Phase 1 Completion Criteria

- [ ] OpenXR runtime functional on at least one device
- [ ] SteamVR runtime functional on at least one device
- [ ] Session state machine handles all lifecycle transitions
- [ ] Guardian boundaries sourced from real SDK data
- [ ] Capability detection validated against hardware
- [ ] Mock runtime supports automated testing

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-1.1: OpenXR Native Bindings | 40 hours |
| T-XR-1.2: SteamVR Integration | 24 hours |
| T-XR-1.3: Session State Machine | 8 hours |
| T-XR-1.4: Capability Detection | 16 hours |
| T-XR-1.5: Guardian SDK Integration | 16 hours |
| T-XR-1.6: Mock Runtime Enhancement | 8 hours |
| T-XR-1.7: Social Services Stubs | 16 hours |
| **Total** | **128 hours** |
