# PHASE 1 ARCHITECTURE: Accessibility Module

## Scope

5 files, ~3,449 lines in `engine/ui/accessibility/`

| File | Lines | Purpose |
|------|-------|---------|
| keyboard_nav.py | 839 | Keyboard navigation and tab order |
| scale.py | 696 | DPI awareness and UI scaling |
| motion.py | 664 | Reduced motion preferences |
| screen_reader.py | 642 | ARIA support and announcements |
| high_contrast.py | 608 | Contrast modes and colorblind support |

---

## Component Architecture

### KeyboardNavigator (keyboard_nav.py)

```
KeyboardNavigator (singleton)
    |
    +-- TabOrder
    |       +-- TabStop[] (ordered focusable elements)
    |
    +-- NavigationGroup[]
    |       +-- Widget[] (grouped navigation targets)
    |
    +-- SkipLink[]
    |       +-- target Widget (accessibility shortcuts)
    |
    +-- KeyboardShortcut[]
            +-- key combo → callback
```

**Key Algorithm**: Spatial navigation finds nearest widget in a direction using bounding box analysis. Calculates center-to-center distance for all candidates and filters by direction vector.

---

### ScaleManager (scale.py)

```
ScaleManager (singleton)
    |
    +-- ScaleConfig
    |       +-- base_dpi: float
    |       +-- min_scale, max_scale: float
    |
    +-- MonitorInfo[]
    |       +-- dpi: float
    |       +-- scale_factor: float
    |
    +-- TouchTargetSize
            +-- MINIMUM_TARGET_SIZE = 44 (WCAG 2.5.5)
```

**Key Invariant**: Touch targets must be at least 44x44 CSS pixels at any scale factor.

---

### MotionManager (motion.py)

```
MotionManager (singleton)
    |
    +-- MotionConfig
    |       +-- preference: AnimationPreference
    |       +-- duration_multiplier: float
    |
    +-- AnimationPreference (enum)
    |       +-- NO_PREFERENCE
    |       +-- REDUCE
    |       +-- NONE
    |
    +-- AnimationCategory (enum)
            +-- ESSENTIAL (feedback, state changes)
            +-- DECORATIVE (parallax, auto-play)
```

**Key Decision**: System preference detection via platform APIs (prefers-reduced-motion on web, accessibility settings on native).

---

### AccessibilityManager (screen_reader.py)

```
AccessibilityManager (singleton)
    |
    +-- AnnouncementQueue
    |       +-- Priority.ASSERTIVE → front of queue
    |       +-- Priority.POLITE → back of queue
    |
    +-- AriaRole (enum, 100+ roles)
    |       +-- BUTTON, CHECKBOX, DIALOG, etc.
    |
    +-- AriaProperty (enum)
    |       +-- describedby, labelledby, controls, etc.
    |
    +-- AriaState (enum)
    |       +-- expanded, selected, checked, disabled, etc.
    |
    +-- AriaLiveRegion
            +-- politeness: POLITE | ASSERTIVE | OFF
```

**Key Algorithm**: Announcement queue prioritizes ASSERTIVE messages (inserted at front) over POLITE (appended to back).

---

### HighContrastManager (high_contrast.py)

```
HighContrastManager (singleton)
    |
    +-- Color (RGBA)
    |       +-- linearize() → sRGB linearization
    |       +-- relative_luminance() → WCAG luminance
    |
    +-- HighContrastTheme (enum)
    |       +-- LIGHT_ON_DARK
    |       +-- DARK_ON_LIGHT
    |       +-- YELLOW_ON_BLACK
    |       +-- CUSTOM
    |
    +-- FocusIndicator
    |       +-- width, color, offset
    |
    +-- ColorblindSimulation
            +-- BRETTEL_PROTAN[3][3]
            +-- BRETTEL_DEUTAN[3][3]
            +-- BRETTEL_TRITAN[3][3]
```

**Critical Algorithms**:

1. **WCAG 2.1 Contrast Ratio**
   - Linearize sRGB components using gamma correction
   - Calculate relative luminance: 0.2126*R + 0.7152*G + 0.0722*B
   - Ratio = (lighter + 0.05) / (darker + 0.05)
   - AA requires 4.5:1 for normal text, 3:1 for large text
   - AAA requires 7:1 for normal text, 4.5:1 for large text

2. **Brettel Colorblind Simulation**
   - Convert RGB to LMS color space
   - Apply transformation matrix for specific deficiency
   - Convert back to RGB
   - Matrices derived from Brettel, Vienot, Mollon (1997)

---

## Data Flow

```
User Input (keyboard)
    |
    v
KeyboardNavigator
    |
    +-- Tab/Shift+Tab → TabOrder.next()/prev()
    +-- Arrow keys → spatial_find_nearest(direction)
    +-- Shortcuts → KeyboardShortcut.invoke()
    |
    v
FocusManager (framework)
    |
    v
AccessibilityManager
    |
    +-- announce_focus_change()
    +-- announce_state_change()
    |
    v
Screen Reader (platform)
```

---

## Integration Points

| From | To | Purpose |
|------|----|---------| 
| KeyboardNavigator | FocusManager | Set focused widget |
| MotionManager | Animation system | Duration multipliers |
| ScaleManager | Layout system | Scale factors |
| HighContrastManager | Render pipeline | Theme colors |
| AccessibilityManager | Platform TTS | Announcements |

---

## Design Decisions

### D1: Singleton Managers

**Decision**: KeyboardNavigator, ScaleManager, MotionManager, AccessibilityManager, HighContrastManager are all singletons.

**Rationale**: Accessibility state is global. A user's preference for reduced motion applies everywhere.

### D2: Category-Based Motion Filtering

**Decision**: Animations are categorized as ESSENTIAL or DECORATIVE.

**Rationale**: ESSENTIAL animations provide necessary feedback (button press, state change). DECORATIVE animations are visual enhancement only (parallax, auto-play video). Reduced motion should disable decorative, not essential.

### D3: WCAG AA as Minimum

**Decision**: Contrast calculations support both AA and AAA thresholds.

**Rationale**: AA is the legal minimum. AAA is the goal. The API exposes both so developers can choose.

### D4: Brettel Over Vienot

**Decision**: Use Brettel (1997) colorblind simulation, not Vienot (1999) simplification.

**Rationale**: Brettel handles anomalous trichromacy (partial colorblindness) more accurately. The extra complexity is justified.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Platform screen reader API differences | Announcements fail on some platforms | Abstract platform layer with fallback |
| Spatial navigation performance | Slow with many widgets | Cache bounding boxes, spatial index |
| Colorblind matrix accuracy | Incorrect simulation | Validate against published test images |
| Scale factor edge cases | Layout breaks at extreme scales | Clamp to min/max range |
