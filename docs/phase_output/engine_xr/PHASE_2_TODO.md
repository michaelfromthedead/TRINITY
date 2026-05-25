# PHASE 2 TODO: Input System and Tracking

## Overview

Phase 2 verifies and hardens the already-implemented input system. The code is production-ready but requires validation against real hardware and integration testing with the runtime layer.

## Tasks

### T-XR-2.1: HMD Tracking Validation

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1 (OpenXR bindings)

**Description**: Validate HMD tracking implementation against real hardware poses from OpenXR runtime.

**Subtasks**:
- [ ] T-XR-2.1.1: Connect HMD component to OpenXR pose data
- [ ] T-XR-2.1.2: Verify position accuracy against known reference points
- [ ] T-XR-2.1.3: Verify orientation accuracy against known rotations
- [ ] T-XR-2.1.4: Validate pose prediction reduces perceived latency
- [ ] T-XR-2.1.5: Test tracking state transitions (TRACKING -> LIMITED -> LOST)

**Acceptance Criteria**:
- [ ] Position accuracy within 1mm at 1m distance
- [ ] Orientation accuracy within 0.5 degrees
- [ ] Pose prediction compensates for at least 20ms
- [ ] Tracking state changes detected within 1 frame

**Files**:
- `engine/xr/input/hmd.py`

---

### T-XR-2.2: Controller Input Testing

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-1.1 or T-XR-1.2

**Description**: Validate controller input processing against real hardware.

**Subtasks**:
- [ ] T-XR-2.2.1: Verify button press/release edge detection
- [ ] T-XR-2.2.2: Verify trigger/grip analog values (0-1 range)
- [ ] T-XR-2.2.3: Verify thumbstick deadzone processing
- [ ] T-XR-2.2.4: Verify capacitive touch detection (if supported)
- [ ] T-XR-2.2.5: Test haptic feedback playback
- [ ] T-XR-2.2.6: Verify grip vs aim pose separation

**Acceptance Criteria**:
- [ ] Button press detected same frame as hardware press
- [ ] Analog values accurate within 1% of full range
- [ ] Deadzone eliminates drift at rest
- [ ] Haptic effects play with correct amplitude/duration
- [ ] Grip pose positions held object correctly

**Files**:
- `engine/xr/input/controller.py`
- `engine/xr/input/haptics.py`

---

### T-XR-2.3: Hand Tracking Integration

**Priority**: High
**Effort**: Large (32 hours)
**Dependencies**: T-XR-1.1 (OpenXR hand tracking extension)

**Description**: Connect hand tracking component to OpenXR hand tracking extension.

**Subtasks**:
- [ ] T-XR-2.3.1: Enable XR_EXT_hand_tracking extension in OpenXR
- [ ] T-XR-2.3.2: Map OpenXR joint indices to component joint indices
- [ ] T-XR-2.3.3: Populate joint positions/orientations/radii
- [ ] T-XR-2.3.4: Validate finger curl calculation accuracy
- [ ] T-XR-2.3.5: Validate pinch detection threshold
- [ ] T-XR-2.3.6: Test gesture recognition on real hand data

**Acceptance Criteria**:
- [ ] All 26 joints populated per frame
- [ ] Finger curl matches visual finger bend
- [ ] Pinch detected when thumb touches index
- [ ] Gestures recognized with >90% accuracy

**Files**:
- `engine/xr/input/hand_tracking.py`

---

### T-XR-2.4: Eye Tracking Integration

**Priority**: High
**Effort**: Large (32 hours)
**Dependencies**: T-XR-1.1 (OpenXR eye tracking extension)

**Description**: Connect eye tracking component to OpenXR eye gaze extension.

**Subtasks**:
- [ ] T-XR-2.4.1: Enable XR_EXT_eye_gaze_interaction extension
- [ ] T-XR-2.4.2: Populate gaze ray from extension data
- [ ] T-XR-2.4.3: Validate fixation detection accuracy
- [ ] T-XR-2.4.4: Validate saccade detection accuracy
- [ ] T-XR-2.4.5: Validate blink detection accuracy
- [ ] T-XR-2.4.6: Test calibration flow on real hardware

**Acceptance Criteria**:
- [ ] Gaze ray intersects looked-at objects
- [ ] Fixation detected when gaze dwells >200ms
- [ ] Saccades detected on rapid gaze shifts
- [ ] Blinks detected with <50ms latency
- [ ] Calibration error <2 degrees post-calibration

**Files**:
- `engine/xr/input/eye_tracking.py`

---

### T-XR-2.5: Action Binding System Testing

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.2

**Description**: Validate action binding system with real controller profiles.

**Subtasks**:
- [ ] T-XR-2.5.1: Create default profile for Quest Touch controllers
- [ ] T-XR-2.5.2: Create default profile for Valve Index controllers
- [ ] T-XR-2.5.3: Test action-to-button binding
- [ ] T-XR-2.5.4: Test action-to-axis binding with threshold
- [ ] T-XR-2.5.5: Test multi-source aggregation
- [ ] T-XR-2.5.6: Test @xr_action decorator integration

**Acceptance Criteria**:
- [ ] Same action code works with both controller types
- [ ] Threshold correctly converts analog to boolean
- [ ] Multiple bindings aggregate correctly
- [ ] Decorator-bound methods called on action

**Files**:
- `engine/xr/input/bindings.py`

---

### T-XR-2.6: Haptic Pattern Library

**Priority**: Medium
**Effort**: Small (8 hours)
**Dependencies**: T-XR-2.2

**Description**: Expand haptic pattern library with XR-standard feedback.

**Subtasks**:
- [ ] T-XR-2.6.1: Add UI interaction patterns (hover, click, drag)
- [ ] T-XR-2.6.2: Add locomotion patterns (teleport, boundary)
- [ ] T-XR-2.6.3: Add object interaction patterns (grab, release, collide)
- [ ] T-XR-2.6.4: Add notification patterns (alert, success, error)
- [ ] T-XR-2.6.5: Test patterns on Quest and Index controllers

**Acceptance Criteria**:
- [ ] UI hover feels distinct from click
- [ ] Teleport activation clearly communicated
- [ ] Object grab feels like physical contact
- [ ] Patterns work across controller types

**Files**:
- `engine/xr/input/haptics.py`

---

### T-XR-2.7: Input Smoothing and Filtering

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.3, T-XR-2.4

**Description**: Add configurable smoothing to reduce tracking jitter.

**Subtasks**:
- [ ] T-XR-2.7.1: Implement one-euro filter for hand joint positions
- [ ] T-XR-2.7.2: Implement exponential smoothing for eye gaze
- [ ] T-XR-2.7.3: Add configurable smoothing strength per use case
- [ ] T-XR-2.7.4: Measure latency impact of filtering
- [ ] T-XR-2.7.5: Test interaction feel with filtered vs raw data

**Acceptance Criteria**:
- [ ] Jitter reduced by >50% at rest
- [ ] Smoothing latency <16ms
- [ ] Configurable per-application

**Files**:
- `engine/xr/input/hand_tracking.py`
- `engine/xr/input/eye_tracking.py`

---

### T-XR-2.8: Input System Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add comprehensive unit tests for input processing logic.

**Subtasks**:
- [ ] T-XR-2.8.1: Test deadzone application
- [ ] T-XR-2.8.2: Test button edge detection
- [ ] T-XR-2.8.3: Test finger curl calculation
- [ ] T-XR-2.8.4: Test gesture recognition logic
- [ ] T-XR-2.8.5: Test fixation detection algorithm
- [ ] T-XR-2.8.6: Test action binding resolution

**Acceptance Criteria**:
- [ ] >90% code coverage on input processing
- [ ] All edge cases tested (zero input, max input, transitions)
- [ ] Tests run without hardware

**Files**:
- `engine/xr/input/tests/` (new directory)

---

## Phase 2 Completion Criteria

- [ ] HMD tracking validated against real hardware
- [ ] Controller input validated on Quest and Index
- [ ] Hand tracking integrated with OpenXR extension
- [ ] Eye tracking integrated with OpenXR extension
- [ ] Action bindings work across controller types
- [ ] Haptic patterns cover common XR interactions
- [ ] Input smoothing reduces jitter without excessive latency
- [ ] Unit tests cover core input processing

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-2.1: HMD Tracking Validation | 16 hours |
| T-XR-2.2: Controller Input Testing | 16 hours |
| T-XR-2.3: Hand Tracking Integration | 32 hours |
| T-XR-2.4: Eye Tracking Integration | 32 hours |
| T-XR-2.5: Action Binding Testing | 16 hours |
| T-XR-2.6: Haptic Pattern Library | 8 hours |
| T-XR-2.7: Input Smoothing | 16 hours |
| T-XR-2.8: Unit Tests | 16 hours |
| **Total** | **152 hours** |
