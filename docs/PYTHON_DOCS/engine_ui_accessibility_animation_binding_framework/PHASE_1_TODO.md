# PHASE 1 TODO: Accessibility Module

## Summary

Verify and test the 5 accessibility files (~3,449 lines) for WCAG compliance and correct implementation.

---

## T1: WCAG Contrast Ratio Validation

**File**: `engine/ui/accessibility/high_contrast.py`

### T1.1: Verify sRGB Linearization Formula
- [ ] Confirm threshold value is exactly 0.03928
- [ ] Confirm divisor is exactly 12.92 for linear region
- [ ] Confirm gamma exponent is exactly 2.4
- [ ] Confirm offset is exactly 0.055

**Acceptance**: Values match WCAG 2.1 specification exactly.

### T1.2: Test Relative Luminance Calculation
- [ ] Test pure white (255,255,255) → luminance = 1.0
- [ ] Test pure black (0,0,0) → luminance = 0.0
- [ ] Test red (255,0,0) → luminance ≈ 0.2126
- [ ] Test green (0,255,0) → luminance ≈ 0.7152
- [ ] Test blue (0,0,255) → luminance ≈ 0.0722

**Acceptance**: All values match expected luminance within 0.0001 tolerance.

### T1.3: Test Contrast Ratio Calculation
- [ ] Test black on white → ratio = 21:1
- [ ] Test white on black → ratio = 21:1 (same)
- [ ] Test known failing pair → ratio < 4.5:1
- [ ] Test known passing pair → ratio >= 4.5:1

**Acceptance**: Ratio calculation matches WCAG examples.

---

## T2: Brettel Colorblind Simulation

**File**: `engine/ui/accessibility/high_contrast.py`

### T2.1: Verify Transformation Matrices
- [ ] Confirm BRETTEL_PROTAN matrix matches published coefficients
- [ ] Confirm BRETTEL_DEUTAN matrix matches published coefficients
- [ ] Confirm BRETTEL_TRITAN matrix matches published coefficients

**Acceptance**: Matrices match Brettel, Vienot, Mollon (1997) paper.

### T2.2: Test Protanopia Simulation
- [ ] Red (255,0,0) should become distinguishable from green
- [ ] Pure gray should remain unchanged
- [ ] Transformation should be reversible for validation

**Acceptance**: Simulated colors match reference implementation.

### T2.3: Test Deuteranopia Simulation
- [ ] Green (0,255,0) should become distinguishable from red
- [ ] Pure gray should remain unchanged

**Acceptance**: Simulated colors match reference implementation.

### T2.4: Test Tritanopia Simulation
- [ ] Blue (0,0,255) should become distinguishable from yellow
- [ ] Pure gray should remain unchanged

**Acceptance**: Simulated colors match reference implementation.

---

## T3: Touch Target Compliance

**File**: `engine/ui/accessibility/scale.py`

### T3.1: Verify Minimum Size Constant
- [ ] Confirm MINIMUM_TARGET_SIZE = 44

**Acceptance**: Matches WCAG 2.5.5 requirement.

### T3.2: Test Size Validation
- [ ] 44x44 should pass
- [ ] 43x44 should fail
- [ ] 44x43 should fail
- [ ] 100x20 should fail (both dimensions must meet minimum)

**Acceptance**: Only targets >= 44 in BOTH dimensions pass.

### T3.3: Test Scale Factor Application
- [ ] At scale 1.0: 44px target is valid
- [ ] At scale 2.0: 88px target is valid (44 CSS pixels)
- [ ] Scaled targets must still meet CSS pixel minimum

**Acceptance**: Scale factors correctly applied to validation.

---

## T4: Reduced Motion System

**File**: `engine/ui/accessibility/motion.py`

### T4.1: Test Preference Levels
- [ ] NO_PREFERENCE allows all animations
- [ ] REDUCE allows ESSENTIAL only
- [ ] NONE allows no animations

**Acceptance**: should_animate() returns correct values for each combination.

### T4.2: Test Duration Multiplier
- [ ] Multiplier = 1.0 → normal speed
- [ ] Multiplier = 2.0 → half speed
- [ ] Multiplier = 0.0 → instant (no animation)

**Acceptance**: Duration multipliers correctly modify animation timing.

### T4.3: Test System Preference Detection
- [ ] Detect platform prefers-reduced-motion setting
- [ ] Fallback to default when platform API unavailable

**Acceptance**: Platform detection works or gracefully degrades.

---

## T5: Screen Reader Integration

**File**: `engine/ui/accessibility/screen_reader.py`

### T5.1: Test ARIA Role Coverage
- [ ] Confirm all standard ARIA roles are defined
- [ ] BUTTON, CHECKBOX, DIALOG, MENU, MENUITEM, etc.

**Acceptance**: AriaRole enum contains 100+ standard roles.

### T5.2: Test Announcement Queue Priority
- [ ] POLITE messages append to queue
- [ ] ASSERTIVE messages insert at front
- [ ] Queue processes in order

**Acceptance**: Priority ordering is correct.

### T5.3: Test Focus Change Announcements
- [ ] Widget gains focus → role + label announced
- [ ] Widget loses focus → no announcement
- [ ] State change on focused widget → state announced

**Acceptance**: Correct announcements at correct times.

### T5.4: Test Live Region Updates
- [ ] POLITE: Announce after current speech
- [ ] ASSERTIVE: Interrupt current speech
- [ ] OFF: No announcement

**Acceptance**: Live region politeness levels work correctly.

---

## T6: Keyboard Navigation

**File**: `engine/ui/accessibility/keyboard_nav.py`

### T6.1: Test Tab Order
- [ ] Tab moves to next TabStop
- [ ] Shift+Tab moves to previous TabStop
- [ ] Tab wraps from last to first
- [ ] Shift+Tab wraps from first to last

**Acceptance**: Tab navigation follows correct order with wrap.

### T6.2: Test Spatial Navigation
- [ ] Arrow Right finds nearest widget to the right
- [ ] Arrow Left finds nearest widget to the left
- [ ] Arrow Down finds nearest widget below
- [ ] Arrow Up finds nearest widget above
- [ ] No valid target → focus unchanged

**Acceptance**: Spatial navigation finds geometrically nearest target.

### T6.3: Test Skip Links
- [ ] Skip to main content works
- [ ] Skip to navigation works
- [ ] Activation moves focus to target

**Acceptance**: Skip links correctly bypass repetitive content.

### T6.4: Test Navigation Groups
- [ ] Group contains related widgets
- [ ] Navigation within group respects boundaries
- [ ] Escape exits group

**Acceptance**: Groups correctly constrain navigation.

### T6.5: Test Keyboard Shortcuts
- [ ] Single key shortcuts work (when focused)
- [ ] Modifier+key shortcuts work (Ctrl+S, etc.)
- [ ] Conflicts are resolved (most specific wins)

**Acceptance**: Shortcut system correctly routes key combinations.

---

## T7: High Contrast Themes

**File**: `engine/ui/accessibility/high_contrast.py`

### T7.1: Test Theme Switching
- [ ] LIGHT_ON_DARK applies white text on black background
- [ ] DARK_ON_LIGHT applies black text on white background
- [ ] YELLOW_ON_BLACK applies yellow text on black background
- [ ] CUSTOM applies user-defined colors

**Acceptance**: Theme colors correctly applied to all widgets.

### T7.2: Test Focus Indicator
- [ ] High-visibility focus ring renders
- [ ] Width is configurable
- [ ] Color is configurable
- [ ] Offset from widget edge is configurable

**Acceptance**: Focus indicator is clearly visible in all themes.

### T7.3: Test Theme Transitions
- [ ] Theme change is smooth (if animations enabled)
- [ ] Theme change is instant (if reduced motion)

**Acceptance**: Theme transitions respect motion preferences.

---

## Completion Criteria

All tasks T1-T7 marked complete with tests passing.
