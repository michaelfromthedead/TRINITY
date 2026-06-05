# Engine XR UI Investigation

**Path:** `engine/xr/ui/`
**Total Lines:** 2,782 (6 files)
**Classification:** REAL - Fully implemented XR UI framework

## Summary

The XR UI module provides a comprehensive, production-ready suite of 3D spatial user interface components for VR/AR/MR applications. This is **not a stub** - all files contain complete implementations with proper state management, interaction handling, and callback systems. The architecture follows XR best practices with support for multiple interaction modes (ray, poke, gaze) and haptic feedback throughout.

## Classification Details

| File | Lines | Status | Implementation Level |
|------|-------|--------|---------------------|
| `__init__.py` | 75 | REAL | Module exports with documentation |
| `panel.py` | 622 | REAL | Full panel system with interaction manager |
| `wrist_ui.py` | 613 | REAL | Complete wrist-mounted UI system |
| `keyboard.py` | 581 | REAL | Virtual keyboard with layouts |
| `slider.py` | 480 | REAL | Interactive slider with haptics |
| `button.py` | 411 | REAL | Button with press depth, haptics |

## XR-Specific Widgets

### 1. XRUIPanel (`panel.py`)

World-space UI container supporting multiple attachment modes:

```python
class XRPanelType(Enum):
    WORLD = auto()         # Fixed in 3D space
    HEAD_LOCKED = auto()   # HUD-style, follows head
    HAND_ATTACHED = auto() # Attached to controller
    WRIST = auto()         # Watch-style interface
```

**Key Features:**
- Configurable dimensions in meters with pixels-per-meter DPI
- Curved display support with configurable radius
- Billboard/face-camera modes
- World-to-panel UV coordinate conversion
- Child element hierarchy management

### 2. XRButton (`button.py`)

Interactive 3D button with physical press simulation:

**Unique Properties:**
- `press_depth`: Physical depth tracking for poke interaction (meters)
- `max_press_depth`: How far button can be pushed (default 0.02m)
- `press_threshold`: Depth required to register press (0.015m)
- `visual_depth`: Returns depth offset for 3D button displacement effect

**Haptic Feedback:**
```python
@dataclass
class HapticFeedback:
    amplitude: float = 0.5    # 0.0-1.0
    duration_ms: int = 50
    frequency: float = 200.0  # Hz
```

Haptic triggers on hover (light), press (configured), and click (release).

### 3. XRSlider (`slider.py`)

Draggable value control for XR:

**Features:**
- Horizontal/vertical orientation
- Value snapping with configurable step
- Normalized value property (0-1)
- Track and handle hit-testing
- Drag callbacks: `on_drag_start`, `on_drag_end`, `on_value_changed`
- Haptic feedback at step boundaries or 10% of range

**XRSliderGroup:** Groups related sliders (e.g., RGB, XYZ) with unified callbacks.

### 4. VirtualKeyboard (`keyboard.py`)

Full virtual keyboard for XR text input:

**Layouts:**
- QWERTY, QWERTY_UPPER
- AZERTY, QWERTZ
- NUMERIC, SYMBOLS, EMOJI

**Key Types:**
```python
class KeyType(Enum):
    CHARACTER, SHIFT, BACKSPACE, ENTER, SPACE,
    TAB, SYMBOLS, ABC, NUMBERS, LANGUAGE,
    HIDE, LEFT, RIGHT, CLEAR, EMOJI
```

**Features:**
- Shift/caps lock toggle (double-tap for caps lock)
- Cursor positioning with LEFT/RIGHT keys
- Text suggestions integration (up to 5)
- Variable key widths (e.g., space = 4.0 units)
- UV-to-key hit detection

### 5. WristUI (`wrist_ui.py`)

Smartwatch-style interface attached to wrist:

**Visibility Modes:**
```python
class WristUIVisibilityMode(Enum):
    ALWAYS = auto()    # Always visible when active
    LOOK_AT = auto()   # Visible when user looks at wrist
    PALM_UP = auto()   # Visible when palm faces up
    MANUAL = auto()    # Manually toggled
```

**Layouts:**
- CIRCULAR: Watch face style (max 8 items)
- RECTANGULAR: 4x3 grid (max 12 items)
- RADIAL: Pie menu slices (max 8 items)

**Integration Points:**
- Pulls config from `engine.xr.config.XR_CONFIG`
- Tracks wrist position/orientation quaternion
- Tracks head position/forward for look-at detection
- Supports notification badges on menu items

**WristUIManager:** Manages left/right wrist UI instances with unified tracking updates.

## 3D UI Architecture

### Coordinate Systems

All dimensions in **meters**:
- Panel: `width=1.0m`, `height=0.75m`, `pixels_per_meter=1000`
- Button: `width=0.15m`, `height=0.05m`
- Slider: `width=0.2m`, `height=0.03m`
- Wrist UI: `size=0.05m` (configurable via XR_CONFIG)
- Keyboard: `width=0.5m`, `height=0.25m`

### Interaction Patterns

#### Ray Interaction (Laser Pointer)
```python
class UIInteractionManager:
    def raycast(origin, direction, interactor_id, max_distance=100.0) -> RaycastHit:
        # Returns: panel, hit_point, uv, distance, normal
```

- Iterates all panels supporting RAY mode
- Performs plane intersection (simplified, assumes -Z facing)
- Returns closest hit with UV coordinates for element lookup
- Updates panel hover state and interactor tracking

#### Poke Interaction (Direct Touch)
```python
def poke(finger_position, finger_id, poke_threshold=0.02) -> PokeInteraction:
    # Returns: panel, touch_point, uv, depth
```

- Checks distance to panel plane
- Button press depth tracked for physical-feel buttons
- Threshold-based press detection

#### Gaze Interaction (Eye Tracking)
```python
def gaze(gaze_origin, gaze_direction, delta_time, user_id) -> GazeInteraction:
    # Returns: panel, gaze_point, uv, dwell_time, is_fixating
```

- Accumulates dwell time on same target
- `is_fixating` becomes True when dwell exceeds threshold (default 1.0s)
- Resets on target change

## Haptic Feedback Integration

All interactive elements support haptic feedback:

| Event | Amplitude | Duration | Frequency |
|-------|-----------|----------|-----------|
| Button hover | 0.1 | 10ms | 150Hz |
| Button press | 0.5 | 50ms | 200Hz |
| Button release | 0.3 | 30ms | 180Hz |
| Slider step | Configurable | Configurable | Configurable |
| Slider continuous | On 10% change | - | - |

## Trinity Integration

All components use Trinity-style decorator metadata:

```python
@xr_ui_panel(panel_type="world", interaction_mode="ray", curved=True)
class MainMenuPanel:
    pass

@xr_button(label="Start", haptic=True, press_depth=0.02)
class StartButton:
    pass

@xr_slider(min_value=0, max_value=100, step=10, haptic=True)
class VolumeSlider:
    pass
```

**Metadata Applied:**
- `_tags`: Dictionary with component metadata
- `_applied_decorators`: Set tracking decorator applications
- `_registries`: Set containing `'xr'` for registry tracking

## External Dependencies

| Dependency | Source | Usage |
|------------|--------|-------|
| `engine.xr.config.XR_CONFIG` | Internal | Wrist UI sizing/thresholds |
| Standard library only | - | math, time, dataclasses, enum |

## Key Observations

### Strengths
1. **Complete Implementation**: No stub code - all methods have functional bodies
2. **Physical Interaction**: Press depth tracking creates physical-feel buttons
3. **Multiple Input Modes**: Ray, poke, and gaze all fully implemented
4. **Haptic Design**: Thoughtful feedback at appropriate moments
5. **Configurability**: Extensive style customization via dataclasses
6. **Performance**: Uses `__slots__` throughout, efficient hit-testing

### Simplifications
1. Panel raycasting assumes panels face -Z direction (noted in comments)
2. Quaternion rotations use simplified math (full implementation would use proper rotation)
3. Curved panel intersection not fully implemented

### Architecture Patterns
- **Dataclass Components**: All UI elements are `@dataclass(slots=True)`
- **Callback Registration**: `on_*` methods for event handling
- **Manager Classes**: `UIInteractionManager`, `KeyboardManager`, `WristUIManager`
- **Group Classes**: `XRButtonGroup`, `XRSliderGroup` for related elements
- **State Tracking**: `_dirty` flags for render optimization

## Integration Points

### With XR Runtime
- Receives tracking data for wrist/head positions
- Panel positions in world space

### With Renderer
- Panel dimensions feed into render texture sizing
- Visual depth offsets for pressed buttons
- Style colors for material generation

### With Input System
- Interactor IDs track which controller/hand is interacting
- Multiple simultaneous interactions supported

## Recommendations

1. **Implement Full Quaternion Math**: Panel world-to-local transforms need proper rotation handling
2. **Add Curved Panel Raycasting**: Currently only planar panels work correctly
3. **Consider Accessibility**: No screen reader or alternative input support yet
4. **Add Animation System**: Button press/release could use spring animations
