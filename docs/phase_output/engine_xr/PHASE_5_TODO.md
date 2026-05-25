# PHASE 5 TODO: Spatial UI System

## Overview

Phase 5 validates and extends the spatial UI system. The implementation is production-ready; this phase focuses on interaction mode testing, haptic tuning, and accessibility verification.

## Tasks

### T-XR-5.1: Ray Interaction Validation

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.2 (controller input)

**Description**: Validate ray casting interaction with UI panels.

**Subtasks**:
- [ ] T-XR-5.1.1: Test ray-plane intersection accuracy
- [ ] T-XR-5.1.2: Test UV coordinate conversion
- [ ] T-XR-5.1.3: Test element hit detection from UV
- [ ] T-XR-5.1.4: Test hover state updates on ray movement
- [ ] T-XR-5.1.5: Test press/release on trigger input
- [ ] T-XR-5.1.6: Test ray visual rendering

**Acceptance Criteria**:
- [ ] Ray intersects panel at correct world position
- [ ] UV coordinates map correctly to elements
- [ ] Hover triggers on element entry/exit
- [ ] Click registers on trigger press

**Files**:
- `engine/xr/ui/panel.py`

---

### T-XR-5.2: Poke Interaction Validation

**Priority**: Critical
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.3 (hand tracking)

**Description**: Validate direct touch (poke) interaction with UI.

**Subtasks**:
- [ ] T-XR-5.2.1: Test proximity detection to panel surface
- [ ] T-XR-5.2.2: Test press depth tracking
- [ ] T-XR-5.2.3: Test press threshold for click registration
- [ ] T-XR-5.2.4: Test visual depth feedback on buttons
- [ ] T-XR-5.2.5: Test haptic feedback on press
- [ ] T-XR-5.2.6: Tune poke_threshold for accuracy vs false positives

**Acceptance Criteria**:
- [ ] Touch detected when finger within 2cm of panel
- [ ] Press depth tracked accurately
- [ ] Click registers at press_threshold depth
- [ ] Button visually depresses with finger

**Files**:
- `engine/xr/ui/button.py`

---

### T-XR-5.3: Gaze Interaction Validation

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.4 (eye tracking)

**Description**: Validate gaze-based selection for accessibility.

**Subtasks**:
- [ ] T-XR-5.3.1: Test gaze ray intersection with panels
- [ ] T-XR-5.3.2: Test dwell time accumulation
- [ ] T-XR-5.3.3: Test dwell threshold for selection
- [ ] T-XR-5.3.4: Test dwell reset on target change
- [ ] T-XR-5.3.5: Add visual dwell progress indicator
- [ ] T-XR-5.3.6: User test gaze selection usability

**Acceptance Criteria**:
- [ ] Gaze targets element under eye focus
- [ ] Dwell time accumulates when fixating
- [ ] Selection triggers at dwell_threshold (1.0s default)
- [ ] Visual progress indicator shows dwell state

**Files**:
- `engine/xr/ui/panel.py`

---

### T-XR-5.4: Button Haptic Tuning

**Priority**: High
**Effort**: Small (8 hours)
**Dependencies**: T-XR-2.6 (haptic patterns)

**Description**: Tune haptic feedback for button interactions.

**Subtasks**:
- [ ] T-XR-5.4.1: Tune hover enter haptic (light tap)
- [ ] T-XR-5.4.2: Tune press haptic (firm click)
- [ ] T-XR-5.4.3: Tune release haptic (confirmation)
- [ ] T-XR-5.4.4: Test haptic on Quest Touch controllers
- [ ] T-XR-5.4.5: Test haptic on Index controllers
- [ ] T-XR-5.4.6: User test haptic feel

**Acceptance Criteria**:
- [ ] Hover haptic noticeable but subtle
- [ ] Press haptic feels like physical button
- [ ] Haptics work on both controller types

**Files**:
- `engine/xr/ui/button.py`

---

### T-XR-5.5: Slider Interaction Tuning

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-5.1, T-XR-5.2

**Description**: Tune slider drag interaction and haptic feedback.

**Subtasks**:
- [ ] T-XR-5.5.1: Test track hit detection
- [ ] T-XR-5.5.2: Test handle drag tracking
- [ ] T-XR-5.5.3: Test value snapping at step intervals
- [ ] T-XR-5.5.4: Test haptic on step boundary
- [ ] T-XR-5.5.5: Test slider group (RGB, XYZ) interaction
- [ ] T-XR-5.5.6: User test slider precision

**Acceptance Criteria**:
- [ ] Slider value tracks finger/controller position
- [ ] Snapping feels natural with step
- [ ] Haptic pulses on step boundaries
- [ ] Slider group updates all values correctly

**Files**:
- `engine/xr/ui/slider.py`

---

### T-XR-5.6: Virtual Keyboard Testing

**Priority**: High
**Effort**: Large (24 hours)
**Dependencies**: T-XR-5.1, T-XR-5.2

**Description**: Test virtual keyboard for accuracy and usability.

**Subtasks**:
- [ ] T-XR-5.6.1: Test key hit detection accuracy
- [ ] T-XR-5.6.2: Test shift/caps lock toggle
- [ ] T-XR-5.6.3: Test backspace and cursor movement
- [ ] T-XR-5.6.4: Test layout switching (QWERTY -> symbols)
- [ ] T-XR-5.6.5: Test suggestion selection
- [ ] T-XR-5.6.6: User test typing speed and error rate

**Acceptance Criteria**:
- [ ] Key presses register on correct character
- [ ] Shift toggles correctly (single tap, double for caps)
- [ ] Cursor moves with LEFT/RIGHT keys
- [ ] Layout switching smooth
- [ ] Typing error rate <10% for experienced users

**Files**:
- `engine/xr/ui/keyboard.py`

---

### T-XR-5.7: Wrist UI Implementation

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-2.3 (hand tracking for wrist position)

**Description**: Implement and test wrist-mounted UI.

**Subtasks**:
- [ ] T-XR-5.7.1: Test wrist position/rotation tracking
- [ ] T-XR-5.7.2: Test LOOK_AT visibility mode
- [ ] T-XR-5.7.3: Test PALM_UP visibility mode
- [ ] T-XR-5.7.4: Test circular layout item selection
- [ ] T-XR-5.7.5: Test rectangular grid layout
- [ ] T-XR-5.7.6: Test notification badges

**Acceptance Criteria**:
- [ ] UI follows wrist position smoothly
- [ ] LOOK_AT triggers when user looks at wrist
- [ ] PALM_UP triggers when palm faces up
- [ ] Items selectable via ray or poke

**Files**:
- `engine/xr/ui/wrist_ui.py`

---

### T-XR-5.8: Curved Panel Implementation

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-5.1

**Description**: Implement correct ray intersection for curved panels.

**Subtasks**:
- [ ] T-XR-5.8.1: Implement ray-cylinder intersection
- [ ] T-XR-5.8.2: Calculate UV on curved surface
- [ ] T-XR-5.8.3: Handle wrap-around (360 degree cylinders)
- [ ] T-XR-5.8.4: Test curved panel rendering
- [ ] T-XR-5.8.5: Test element hit detection on curves

**Acceptance Criteria**:
- [ ] Ray intersects curved panel correctly
- [ ] UV maps correctly around curve
- [ ] Elements remain interactive on curved surface

**Files**:
- `engine/xr/ui/panel.py`

---

### T-XR-5.9: UI Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add unit tests for UI interaction logic.

**Subtasks**:
- [ ] T-XR-5.9.1: Test ray-plane intersection math
- [ ] T-XR-5.9.2: Test UV-to-element mapping
- [ ] T-XR-5.9.3: Test dwell time accumulation
- [ ] T-XR-5.9.4: Test keyboard key lookup
- [ ] T-XR-5.9.5: Test visibility mode transitions

**Acceptance Criteria**:
- [ ] >85% code coverage on UI core
- [ ] Intersection math verified geometrically
- [ ] State transitions verified exhaustively

**Files**:
- `engine/xr/ui/tests/` (new directory)

---

## Phase 5 Completion Criteria

- [ ] Ray interaction validated with controller
- [ ] Poke interaction validated with hand tracking
- [ ] Gaze interaction validated with eye tracking
- [ ] Button haptics tuned for natural feel
- [ ] Slider interaction precise and haptic
- [ ] Virtual keyboard usable for text input
- [ ] Wrist UI functional with visibility modes
- [ ] Curved panels support ray intersection
- [ ] Unit tests cover UI interaction logic

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-5.1: Ray Interaction | 16 hours |
| T-XR-5.2: Poke Interaction | 16 hours |
| T-XR-5.3: Gaze Interaction | 16 hours |
| T-XR-5.4: Button Haptics | 8 hours |
| T-XR-5.5: Slider Interaction | 16 hours |
| T-XR-5.6: Virtual Keyboard | 24 hours |
| T-XR-5.7: Wrist UI | 16 hours |
| T-XR-5.8: Curved Panels | 16 hours |
| T-XR-5.9: Unit Tests | 16 hours |
| **Total** | **144 hours** |
