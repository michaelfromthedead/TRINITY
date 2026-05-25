# PHASE 3 TODO: Avatar System

## Overview

Phase 3 validates and extends the avatar system. The core implementation is production-ready; this phase focuses on integration with input systems, network validation, and user experience polish.

## Tasks

### T-XR-3.1: IK Solver Validation

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.1 (HMD tracking), T-XR-2.2 (controller input)

**Description**: Validate IK solvers produce correct joint positions from tracked input.

**Subtasks**:
- [ ] T-XR-3.1.1: Test TwoBone solver with arm chain
- [ ] T-XR-3.1.2: Test TwoBone solver with leg chain
- [ ] T-XR-3.1.3: Test pole target positioning (elbow direction)
- [ ] T-XR-3.1.4: Test FABRIK with 5+ joint chain
- [ ] T-XR-3.1.5: Test CCD with joint angle limits
- [ ] T-XR-3.1.6: Profile solver performance at 90Hz

**Acceptance Criteria**:
- [ ] Arm reaches target position within 1cm tolerance
- [ ] Elbow bends in pole target direction
- [ ] FABRIK converges in <10 iterations
- [ ] CCD respects joint angle limits
- [ ] All solvers complete in <0.5ms

**Files**:
- `engine/xr/avatars/ik_solver.py`

---

### T-XR-3.2: Hand Animation from Tracking

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.3 (hand tracking)

**Description**: Connect hand tracking data to hand animation system.

**Subtasks**:
- [ ] T-XR-3.2.1: Map hand tracking joints to finger curl values
- [ ] T-XR-3.2.2: Verify curl calculation matches visual finger bend
- [ ] T-XR-3.2.3: Test pose blending during transitions
- [ ] T-XR-3.2.4: Test controller fallback when hand tracking lost
- [ ] T-XR-3.2.5: Validate grip/pinch strength metrics

**Acceptance Criteria**:
- [ ] Avatar hand visually matches physical hand position
- [ ] Pose transitions smooth over 0.1-0.2 seconds
- [ ] Controller-to-hand-tracking transition seamless
- [ ] Grip strength maps to grab action

**Files**:
- `engine/xr/avatars/hand_animator.py`

---

### T-XR-3.3: Face Tracking Integration

**Priority**: High
**Effort**: Large (24 hours)
**Dependencies**: T-XR-2.4 (eye tracking)

**Description**: Connect eye tracking and future face tracking to blend shape system.

**Subtasks**:
- [ ] T-XR-3.3.1: Map eye gaze to look blend shapes
- [ ] T-XR-3.3.2: Map eye openness to blink blend shapes
- [ ] T-XR-3.3.3: Test auto-blink when eye tracking unavailable
- [ ] T-XR-3.3.4: Validate expression preset application
- [ ] T-XR-3.3.5: Prepare hooks for future face tracking extension
- [ ] T-XR-3.3.6: Test lip sync viseme mapping

**Acceptance Criteria**:
- [ ] Avatar eyes follow physical eye gaze
- [ ] Blinks render naturally
- [ ] Auto-blink activates every 3-5 seconds
- [ ] Expression presets apply correct blend shapes

**Files**:
- `engine/xr/avatars/face_tracking.py`

---

### T-XR-3.4: Calibration Flow Testing

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.1

**Description**: Test and polish the calibration user experience.

**Subtasks**:
- [ ] T-XR-3.4.1: Test floor detection accuracy
- [ ] T-XR-3.4.2: Test height measurement accuracy
- [ ] T-XR-3.4.3: Test arm span measurement in T-pose
- [ ] T-XR-3.4.4: Test quick calibration vs guided calibration accuracy
- [ ] T-XR-3.4.5: Test calibration persistence (save/load)
- [ ] T-XR-3.4.6: Add visual/audio feedback during calibration

**Acceptance Criteria**:
- [ ] Floor level accurate within 2cm
- [ ] Height accurate within 3cm
- [ ] Arm span accurate within 5cm
- [ ] Calibration persists across sessions
- [ ] User understands each calibration step

**Files**:
- `engine/xr/avatars/calibration.py`

---

### T-XR-3.5: Body Estimation Tuning

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-3.4

**Description**: Tune body estimation to look natural with only three tracked points.

**Subtasks**:
- [ ] T-XR-3.5.1: Test pelvis positioning at various heights
- [ ] T-XR-3.5.2: Test chest interpolation during bending
- [ ] T-XR-3.5.3: Test procedural foot placement during walking
- [ ] T-XR-3.5.4: Tune stride width per calibration
- [ ] T-XR-3.5.5: Add locomotion prediction for foot animation

**Acceptance Criteria**:
- [ ] Body looks natural during standing
- [ ] Body looks natural during crouching
- [ ] Feet animate convincingly during walking
- [ ] No visible joint popping

**Files**:
- `engine/xr/avatars/avatar.py`

---

### T-XR-3.6: Personal Space Implementation

**Priority**: Medium
**Effort**: Small (8 hours)
**Dependencies**: None

**Description**: Implement personal space enforcement for multiplayer.

**Subtasks**:
- [ ] T-XR-3.6.1: Test invasion detection at configured radius
- [ ] T-XR-3.6.2: Test push vector generation
- [ ] T-XR-3.6.3: Test fade alpha calculation
- [ ] T-XR-3.6.4: Test visual boundary indicator
- [ ] T-XR-3.6.5: Add configurable personal space radius

**Acceptance Criteria**:
- [ ] Invasion detected when other avatar <0.5m
- [ ] Push vector direction is away from invader
- [ ] Invading avatar fades at boundary
- [ ] Boundary ring visible when invaded

**Files**:
- `engine/xr/avatars/avatar.py`

---

### T-XR-3.7: Network Sync Validation

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-3.1 through T-XR-3.6

**Description**: Validate network serialization for multiplayer avatar sync.

**Subtasks**:
- [ ] T-XR-3.7.1: Measure get_network_state() bandwidth per avatar
- [ ] T-XR-3.7.2: Test apply_network_state() reconstruction accuracy
- [ ] T-XR-3.7.3: Test delta compression for static avatars
- [ ] T-XR-3.7.4: Test face tracking bandwidth optimization
- [ ] T-XR-3.7.5: Simulate 30Hz sync with 100ms latency

**Acceptance Criteria**:
- [ ] Full avatar state <200 bytes
- [ ] Face state <80 bytes (non-zero shapes only)
- [ ] Reconstruction error <1cm position, <2deg rotation
- [ ] Avatar looks natural with 30Hz sync

**Files**:
- `engine/xr/avatars/avatar.py`
- `engine/xr/avatars/hand_animator.py`
- `engine/xr/avatars/face_tracking.py`

---

### T-XR-3.8: Avatar Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add unit tests for avatar system components.

**Subtasks**:
- [ ] T-XR-3.8.1: Test IK solver math (TwoBone law of cosines)
- [ ] T-XR-3.8.2: Test hand pose interpolation
- [ ] T-XR-3.8.3: Test blend shape weight clamping
- [ ] T-XR-3.8.4: Test calibration proportion calculations
- [ ] T-XR-3.8.5: Test personal space distance calculations

**Acceptance Criteria**:
- [ ] >85% code coverage on avatar core
- [ ] IK solver edge cases tested
- [ ] Calibration math verified against human proportions

**Files**:
- `engine/xr/avatars/tests/` (new directory)

---

## Phase 3 Completion Criteria

- [ ] IK solvers produce correct positions from tracked input
- [ ] Hand animation matches hand tracking data
- [ ] Face tracking drives blend shapes correctly
- [ ] Calibration flow guides user through measurements
- [ ] Body estimation looks natural with three-point tracking
- [ ] Personal space enforced in multiplayer
- [ ] Network sync validated for bandwidth and accuracy
- [ ] Unit tests cover core avatar logic

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-3.1: IK Solver Validation | 16 hours |
| T-XR-3.2: Hand Animation | 16 hours |
| T-XR-3.3: Face Tracking | 24 hours |
| T-XR-3.4: Calibration Flow | 16 hours |
| T-XR-3.5: Body Estimation | 16 hours |
| T-XR-3.6: Personal Space | 8 hours |
| T-XR-3.7: Network Sync | 16 hours |
| T-XR-3.8: Unit Tests | 16 hours |
| **Total** | **128 hours** |
