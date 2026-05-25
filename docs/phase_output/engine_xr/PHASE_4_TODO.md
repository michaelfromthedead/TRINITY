# PHASE 4 TODO: Locomotion and Comfort

## Overview

Phase 4 integrates and validates the locomotion system. The core implementation is complete; this phase focuses on physics integration, comfort tuning, and user testing.

## Tasks

### T-XR-4.1: Teleport Physics Integration

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.2 (controller input)

**Description**: Integrate teleport arc calculation with collision system.

**Subtasks**:
- [ ] T-XR-4.1.1: Connect arc endpoint to raycast for ground detection
- [ ] T-XR-4.1.2: Validate arc-to-wall collision blocking
- [ ] T-XR-4.1.3: Test max_distance arc cutoff
- [ ] T-XR-4.1.4: Test snap rotation at destination
- [ ] T-XR-4.1.5: Test FADE vs INSTANT vs DASH transitions
- [ ] T-XR-4.1.6: Add visual arc rendering integration

**Acceptance Criteria**:
- [ ] Arc lands on walkable surfaces only
- [ ] Arc blocked by walls and obstacles
- [ ] Teleport completes within fade_duration
- [ ] Snap rotation aligns to surface or fixed angle

**Files**:
- `engine/xr/locomotion/teleport.py`

---

### T-XR-4.2: Smooth Locomotion Tuning

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.2

**Description**: Tune smooth locomotion input processing for natural feel.

**Subtasks**:
- [ ] T-XR-4.2.1: Tune deadzone to eliminate drift
- [ ] T-XR-4.2.2: Tune input curve for precision at low speeds
- [ ] T-XR-4.2.3: Test head vs hand direction modes
- [ ] T-XR-4.2.4: Test snap turn angle and cooldown
- [ ] T-XR-4.2.5: Test smooth turn speed and deceleration
- [ ] T-XR-4.2.6: Validate backward and strafe speed ratios

**Acceptance Criteria**:
- [ ] No drift at thumbstick rest position
- [ ] Precise control at slow walk speeds
- [ ] Direction feels natural for head/hand modes
- [ ] Snap turn does not cause discomfort

**Files**:
- `engine/xr/locomotion/smooth.py`

---

### T-XR-4.3: Climbing System Integration

**Priority**: High
**Effort**: Large (24 hours)
**Dependencies**: T-XR-2.2, T-XR-2.3 (hand tracking optional)

**Description**: Integrate climbing with physics and validate state machine.

**Subtasks**:
- [ ] T-XR-4.3.1: Test grab detection on climbable volumes
- [ ] T-XR-4.3.2: Test climb movement = inverse hand movement
- [ ] T-XR-4.3.3: Test dual-hand climbing (average velocity)
- [ ] T-XR-4.3.4: Test stamina drain and recovery
- [ ] T-XR-4.3.5: Test mantle detection and animation
- [ ] T-XR-4.3.6: Add haptic feedback on grab/release

**Acceptance Criteria**:
- [ ] Grab activates at grip_threshold
- [ ] Player moves opposite to hand pull direction
- [ ] Stamina drains while climbing, recovers on ground
- [ ] Mantle triggers at ledge top

**Files**:
- `engine/xr/locomotion/climbing.py`

---

### T-XR-4.4: Comfort Vignette Calibration

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-4.2

**Description**: Calibrate vignette thresholds and fade behavior.

**Subtasks**:
- [ ] T-XR-4.4.1: Tune velocity threshold for activation
- [ ] T-XR-4.4.2: Tune angular threshold for activation
- [ ] T-XR-4.4.3: Tune fade in/out speeds
- [ ] T-XR-4.4.4: Test circular vs elliptical vs rectangular shapes
- [ ] T-XR-4.4.5: Validate shader integration
- [ ] T-XR-4.4.6: User test vignette comfort effectiveness

**Acceptance Criteria**:
- [ ] Vignette activates before discomfort onset
- [ ] Fade transitions do not cause discomfort
- [ ] Intensity reduces motion sickness in 80% of users
- [ ] Shader renders correctly at all FOV angles

**Files**:
- `engine/xr/locomotion/comfort.py`

---

### T-XR-4.5: Comfort Preset Validation

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-4.4

**Description**: Validate comfort presets against user comfort profiles.

**Subtasks**:
- [ ] T-XR-4.5.1: Test "veteran" preset (all comfort off)
- [ ] T-XR-4.5.2: Test "comfortable" preset (most users)
- [ ] T-XR-4.5.3: Test "maximum" preset (sensitive users)
- [ ] T-XR-4.5.4: Test "seated" preset (seated play)
- [ ] T-XR-4.5.5: Test preset persistence (save/load)
- [ ] T-XR-4.5.6: User test preset appropriateness

**Acceptance Criteria**:
- [ ] Presets feel distinct from each other
- [ ] "comfortable" works for typical new VR users
- [ ] "maximum" eliminates discomfort for sensitive users
- [ ] Presets persist across sessions

**Files**:
- `engine/xr/locomotion/comfort.py`

---

### T-XR-4.6: Arm Swing Locomotion

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.2

**Description**: Implement and tune arm swing movement mode.

**Subtasks**:
- [ ] T-XR-4.6.1: Detect arm swing from controller velocity
- [ ] T-XR-4.6.2: Convert swing frequency to movement speed
- [ ] T-XR-4.6.3: Determine direction from arm movement pattern
- [ ] T-XR-4.6.4: Add minimum swing threshold to prevent drift
- [ ] T-XR-4.6.5: User test arm swing feel

**Acceptance Criteria**:
- [ ] Walking arm motion produces forward movement
- [ ] Speed proportional to swing frequency
- [ ] No movement when arms stationary
- [ ] Feels natural to users

**Files**:
- `engine/xr/locomotion/smooth.py`

---

### T-XR-4.7: Locomotion Provider Integration

**Priority**: Medium
**Effort**: Small (8 hours)
**Dependencies**: T-XR-4.1 through T-XR-4.3

**Description**: Validate provider pattern for runtime integration.

**Subtasks**:
- [ ] T-XR-4.7.1: Test switching between teleport and smooth
- [ ] T-XR-4.7.2: Test hybrid locomotion (teleport + smooth)
- [ ] T-XR-4.7.3: Test climbing override of smooth locomotion
- [ ] T-XR-4.7.4: Validate enabled/disabled state per provider

**Acceptance Criteria**:
- [ ] Mode switching seamless at runtime
- [ ] Hybrid modes combine correctly
- [ ] Climbing takes priority when grabbing

**Files**:
- `engine/xr/locomotion/__init__.py`

---

### T-XR-4.8: Locomotion Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add unit tests for locomotion calculations.

**Subtasks**:
- [ ] T-XR-4.8.1: Test arc projectile physics
- [ ] T-XR-4.8.2: Test deadzone application
- [ ] T-XR-4.8.3: Test input curve transformation
- [ ] T-XR-4.8.4: Test climbing state transitions
- [ ] T-XR-4.8.5: Test vignette intensity calculation
- [ ] T-XR-4.8.6: Test comfort preset application

**Acceptance Criteria**:
- [ ] >85% code coverage on locomotion core
- [ ] Physics calculations verified mathematically
- [ ] State machine transitions verified exhaustively

**Files**:
- `engine/xr/locomotion/tests/` (new directory)

---

## Phase 4 Completion Criteria

- [ ] Teleport integrated with collision system
- [ ] Smooth locomotion tuned for natural feel
- [ ] Climbing state machine validated
- [ ] Vignette calibrated for comfort effectiveness
- [ ] Comfort presets validated with users
- [ ] Arm swing mode functional
- [ ] Provider pattern enables mode switching
- [ ] Unit tests cover core locomotion logic

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-4.1: Teleport Physics | 16 hours |
| T-XR-4.2: Smooth Locomotion | 16 hours |
| T-XR-4.3: Climbing System | 24 hours |
| T-XR-4.4: Vignette Calibration | 16 hours |
| T-XR-4.5: Preset Validation | 16 hours |
| T-XR-4.6: Arm Swing | 16 hours |
| T-XR-4.7: Provider Integration | 8 hours |
| T-XR-4.8: Unit Tests | 16 hours |
| **Total** | **128 hours** |
