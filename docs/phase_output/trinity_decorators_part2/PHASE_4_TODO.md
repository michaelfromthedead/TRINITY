# PHASE 4 TODO: Animation, Audio & Presentation

## Overview

Validate IK, animation, cinematics, audio, UI, localization, and accessibility decorators.

---

## T4.1: Validate IK Procedural Decorators

**File**: `trinity/decorators/ik_procedural.py`

**Tasks**:
- [ ] Verify `@ik_chain` validates solver against VALID_IK_SOLVERS
- [ ] Verify `@ik_chain` accepts joint configuration
- [ ] Verify `@ik_goal` sets IK target
- [ ] Verify `@procedural_bone` marks procedural animation
- [ ] Verify `@motion_matching` validates features
- [ ] Verify `@ragdoll` sets physics parameters

**IK Solvers**:
- [ ] "fabrik" - Forward And Backward Reaching IK
- [ ] "ccd" - Cyclic Coordinate Descent
- [ ] "jacobian" - Jacobian-based
- [ ] "analytical" - Analytical solution

**Motion Matching Features**:
- [ ] "position" - joint positions
- [ ] "velocity" - joint velocities
- [ ] "trajectory" - future trajectory
- [ ] "pose" - full pose matching

**Acceptance Criteria**:
- All 5 decorators follow 6-part pattern
- Solver validation produces actionable error
- All register in "animation" registry

---

## T4.2: Validate Animation Decorators

**File**: `trinity/decorators/animation.py`

**Tasks**:
- [ ] Verify `@tween` validates easing function
- [ ] Verify `@tween` accepts duration parameter
- [ ] Verify `@blend_tree` configures blend nodes
- [ ] Verify all easing functions in VALID_EASING

**Easing Functions** (partial list):
- [ ] "linear"
- [ ] "ease_in", "ease_out", "ease_in_out"
- [ ] "ease_in_quad", "ease_out_quad", "ease_in_out_quad"
- [ ] "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic"
- [ ] "ease_in_elastic", "ease_out_elastic", "ease_in_out_elastic"
- [ ] "ease_in_bounce", "ease_out_bounce", "ease_in_out_bounce"

**Acceptance Criteria**:
- Both decorators validate easing functions
- Invalid easing produces error with valid options listed
- Blend tree accepts node configuration

---

## T4.3: Validate Cinematics Decorators

**File**: `trinity/decorators/cinematics.py`

**Tasks**:
- [ ] Verify `@cutscene` sets timeline metadata
- [ ] Verify `@camera_track` validates blend type
- [ ] Verify camera blends are in VALID_CAMERA_BLENDS

**Camera Blend Types**:
- [ ] "cut" - instant switch
- [ ] "ease" - eased transition
- [ ] "smooth" - smooth interpolation
- [ ] "spring" - spring-based blend

**Acceptance Criteria**:
- Both decorators produce correct steps
- Blend validation produces actionable error
- All register in "cinematics" registry

---

## T4.4: Validate Audio Decorators

**File**: `trinity/decorators/audio.py`

**Tasks**:
- [ ] Verify `@sound` accepts sound asset reference
- [ ] Verify `@audio_bus` configures bus routing
- [ ] Verify `@spatial_audio` validates falloff curve
- [ ] Verify falloff curves in VALID_FALLOFF

**Falloff Curves**:
- [ ] "linear" - linear falloff
- [ ] "logarithmic" - log falloff (realistic)
- [ ] "inverse_square" - physics-based
- [ ] "custom" - user-defined curve

**Acceptance Criteria**:
- All 3 decorators follow 6-part pattern
- Falloff validation produces actionable error
- All register in "audio" registry

---

## T4.5: Validate UI Decorators

**File**: `trinity/decorators/ui.py`

**Tasks**:
- [ ] Verify `@widget` marks UI component
- [ ] Verify `@layout` validates direction
- [ ] Verify directions in VALID_DIRECTIONS

**Layout Directions**:
- [ ] "horizontal" - left to right
- [ ] "vertical" - top to bottom
- [ ] "grid" - grid layout
- [ ] "flow" - flow layout

**Acceptance Criteria**:
- Both decorators produce correct steps
- Direction validation produces actionable error
- All register in "ui" registry

---

## T4.6: Validate Localization Decorators

**File**: `trinity/decorators/localization.py`

**Tasks**:
- [ ] Verify `@localized` marks text for translation
- [ ] Verify `@plural` configures pluralization rules
- [ ] Verify `@rtl_aware` enables RTL support
- [ ] Verify `@text_overflow` sets overflow handling
- [ ] Verify `_validate_target_type` checks for text fields

**Target Type Validation**:
```python
@localized  # Applied to non-text field should raise:
# TypeError: @localized can only be applied to text fields
```

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Target type validation enforced
- All register in "localization" registry

---

## T4.7: Validate Accessibility Decorator

**File**: `trinity/decorators/accessibility.py`

**Tasks**:
- [ ] Verify `@accessible` validates role against VALID_ROLES
- [ ] Verify ARIA roles follow W3C specifications
- [ ] Verify screen reader hints can be set
- [ ] Verify keyboard navigation support

**ARIA Roles** (partial list):
- [ ] "button", "checkbox", "dialog"
- [ ] "grid", "heading", "img", "link"
- [ ] "list", "listitem", "menu", "menuitem"
- [ ] "navigation", "progressbar", "radiogroup"
- [ ] "slider", "spinbutton", "tab", "tablist"
- [ ] "tabpanel", "textbox", "tree", "treeitem"

**Acceptance Criteria**:
- Role validation produces actionable error
- ARIA roles are W3C compliant
- Registers in "accessibility" registry

---

## Summary

| Task | File | Decorators | Lines |
|------|------|------------|-------|
| T4.1 | ik_procedural.py | 5 | 288 |
| T4.2 | animation.py | 2 | 160 |
| T4.3 | cinematics.py | 2 | 155 |
| T4.4 | audio.py | 3 | 218 |
| T4.5 | ui.py | 2 | 165 |
| T4.6 | localization.py | 4 | 227 |
| T4.7 | accessibility.py | 1 | 120 |

**Total**: 19 decorators, 1,333 lines
