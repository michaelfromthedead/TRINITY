# PHASE 5 ARCHITECTURE: Spatial UI System

## Phase Overview

Phase 5 implements the 3D spatial user interface system for XR applications. This phase covers world-space panels, interactive widgets (buttons, sliders), virtual keyboard, and wrist-mounted UI. The UI system must support three interaction modes: ray casting, direct touch (poke), and gaze-based selection.

## Architectural Decisions

### ADR-XR-040: World-Space UI Coordinate System

**Context**: UI elements exist in 3D world space but need 2D layout and hit testing.

**Decision**: Use meters as the primary unit with configurable pixels-per-meter:
- Panel dimensions: width/height in meters (e.g., 1.0m x 0.75m)
- Pixels per meter: resolution density (e.g., 1000 ppm = 1000x750 panel)
- UV coordinates: normalized 0-1 for hit testing

**Consequences**:
- Panels scale naturally with world
- Consistent DPI across viewing distances
- Simple world-to-UV conversion

### ADR-XR-041: Multi-Modal Interaction Strategy

**Context**: Users interact with XR UI via ray, poke, or gaze; all must work seamlessly.

**Decision**: Implement three interaction modes with unified result:
1. **Ray**: Laser pointer from controller, intersects panel plane
2. **Poke**: Direct finger touch, proximity-based hit detection
3. **Gaze**: Eye tracking with dwell time for selection

All return same hit data: panel, UV coordinate, depth (for poke).

**Consequences**:
- Same widgets work with all input modes
- Mode switching transparent to application
- Accessibility via gaze for users with motor limitations

### ADR-XR-042: Physical Button Press Model

**Context**: XR buttons should feel physical, not flat like 2D buttons.

**Decision**: Track press depth for poke interaction:
- `press_depth`: How far finger has pushed button (meters)
- `max_press_depth`: Button travel distance (default 2cm)
- `press_threshold`: Depth for click registration (default 1.5cm)
- `visual_depth`: Z offset for rendering pressed state

**Consequences**:
- Buttons feel like physical switches
- Haptic feedback tied to press depth
- Visual feedback matches physical state

### ADR-XR-043: Haptic Feedback Integration

**Context**: Every UI interaction should provide haptic confirmation.

**Decision**: Define haptic events per interaction:
| Event | Amplitude | Duration | Frequency |
|-------|-----------|----------|-----------|
| Hover Enter | 0.1 | 10ms | 150Hz |
| Button Press | 0.5 | 50ms | 200Hz |
| Button Release | 0.3 | 30ms | 180Hz |
| Slider Step | configurable | configurable | configurable |

**Consequences**:
- Interactions feel responsive
- Users can interact with eyes closed
- Haptic strength configurable per user

### ADR-XR-044: Virtual Keyboard Architecture

**Context**: Text input in XR requires virtual keyboard with layout flexibility.

**Decision**: Implement keyboard as panel with key grid:
- Multiple layouts: QWERTY, AZERTY, QWERTZ, numeric, symbols
- Special keys: SHIFT, CAPS, BACKSPACE, ENTER, SPACE
- Suggestions bar: up to 5 word completions
- Variable key widths: space bar = 4x normal key

**Consequences**:
- International layout support
- Predictive text possible with suggestions
- Standard typing UX

### ADR-XR-045: Wrist UI Design

**Context**: Users need quick-access UI without looking away from scene.

**Decision**: Implement smartwatch-style wrist UI:
- Attached to wrist joint position
- Visibility modes: ALWAYS, LOOK_AT, PALM_UP, MANUAL
- Layouts: CIRCULAR (watch face), RECTANGULAR (grid), RADIAL (pie menu)
- Max items: 8 (circular), 12 (rectangular), 8 (radial)

**Consequences**:
- Quick access to common actions
- Natural gesture activation (look at wrist, palm up)
- Familiar smartwatch paradigm

### ADR-XR-046: Panel Attachment Modes

**Context**: Panels need different attachment behaviors for different uses.

**Decision**: Implement four attachment modes:
1. **WORLD**: Fixed in 3D space (menus, signs)
2. **HEAD_LOCKED**: HUD-style, follows head with lag (status display)
3. **HAND_ATTACHED**: Attached to controller (tool menus)
4. **WRIST**: Watch-style on wrist (quick actions)

**Consequences**:
- Right attachment mode for each use case
- Head-locked panels stay comfortable with lag
- Hand panels accessible without looking

## Component Specifications

### XR Panel System

```
XRUIPanel (Component)
├── Configuration
│   ├── width: float (meters)
│   ├── height: float (meters)
│   ├── pixels_per_meter: float (default 1000)
│   ├── panel_type: XRPanelType (WORLD, HEAD_LOCKED, HAND_ATTACHED, WRIST)
│   ├── interaction_mode: InteractionMode (RAY, POKE, GAZE, ALL)
│   ├── is_curved: bool
│   ├── curve_radius: float
│   └── billboard: bool (always face camera)
├── Positioning
│   ├── position: Vec3 (world space)
│   ├── rotation: Quat (world space)
│   └── calculate_world_to_uv(world_point) -> Vec2
├── Child Elements
│   ├── elements: List[XRUIElement]
│   ├── add_element(element) -> None
│   ├── remove_element(element) -> None
│   └── get_element_at_uv(uv) -> Optional[XRUIElement]
├── State
│   ├── is_visible: bool
│   ├── is_interactable: bool
│   ├── hover_interactor_id: Optional[str]
│   └── focus_element: Optional[XRUIElement]
└── Render
    ├── is_dirty: bool
    └── resolution: Tuple[int, int]

XRPanelType Enum
├── WORLD
├── HEAD_LOCKED
├── HAND_ATTACHED
└── WRIST

InteractionMode Enum
├── RAY
├── POKE
├── GAZE
└── ALL
```

### UI Interaction Manager

```
UIInteractionManager (Singleton)
├── Panel Registry
│   ├── register_panel(panel) -> None
│   ├── unregister_panel(panel) -> None
│   └── get_panels_by_mode(mode) -> List[XRUIPanel]
├── Ray Interaction
│   └── raycast(origin, direction, interactor_id, max_distance) -> RaycastHit
├── Poke Interaction
│   └── poke(finger_position, finger_id, poke_threshold) -> PokeInteraction
├── Gaze Interaction
│   └── gaze(gaze_origin, gaze_direction, delta_time, user_id) -> GazeInteraction
└── State Updates
    ├── update_hover(panel, uv, interactor_id) -> None
    ├── update_press(panel, uv, interactor_id) -> None
    └── update_release(panel, interactor_id) -> None

RaycastHit
├── panel: XRUIPanel
├── hit_point: Vec3
├── uv: Vec2
├── distance: float
└── normal: Vec3

PokeInteraction
├── panel: XRUIPanel
├── touch_point: Vec3
├── uv: Vec2
├── depth: float
└── is_touching: bool

GazeInteraction
├── panel: XRUIPanel
├── gaze_point: Vec3
├── uv: Vec2
├── dwell_time: float
├── is_fixating: bool
└── dwell_threshold: float (default 1.0s)
```

### XR Button

```
XRButton (XRUIElement)
├── Configuration
│   ├── width: float (meters)
│   ├── height: float (meters)
│   ├── label: str
│   ├── icon: Optional[str]
│   └── haptic: HapticFeedback
├── Press State
│   ├── is_hovered: bool
│   ├── is_pressed: bool
│   ├── press_depth: float (meters, for poke)
│   ├── max_press_depth: float (default 0.02m)
│   └── press_threshold: float (default 0.015m)
├── Callbacks
│   ├── on_hover_enter: Callable
│   ├── on_hover_exit: Callable
│   ├── on_press: Callable
│   ├── on_release: Callable
│   └── on_click: Callable
├── Visual State
│   ├── visual_depth: float (Z offset for rendering)
│   └── style: ButtonStyle (colors, border, etc.)
└── Hit Testing
    └── contains_point(local_position) -> bool

XRButtonGroup
├── buttons: List[XRButton]
├── selection_mode: SelectionMode (SINGLE, MULTI)
├── selected_index: int
└── on_selection_changed: Callable

HapticFeedback
├── amplitude: float (0-1)
├── duration_ms: int
└── frequency: float (Hz)

@xr_button Decorator
└── Configures button with label, haptic, press_depth
```

### XR Slider

```
XRSlider (XRUIElement)
├── Configuration
│   ├── width: float (meters)
│   ├── height: float (meters)
│   ├── orientation: SliderOrientation (HORIZONTAL, VERTICAL)
│   ├── min_value: float
│   ├── max_value: float
│   ├── step: Optional[float] (for snapping)
│   └── haptic: SliderHaptic
├── State
│   ├── value: float
│   ├── normalized_value: float (0-1)
│   ├── is_dragging: bool
│   └── drag_interactor_id: Optional[str]
├── Callbacks
│   ├── on_value_changed: Callable[[float], None]
│   ├── on_drag_start: Callable
│   └── on_drag_end: Callable
├── Hit Testing
│   ├── track_bounds: Rect
│   ├── handle_bounds: Rect
│   └── handle_position: float (0-1)
└── Haptic
    ├── haptic_on_step: bool
    └── haptic_on_change: bool (10% increments)

XRSliderGroup
├── sliders: List[XRSlider]
├── labels: List[str] (e.g., "R", "G", "B")
└── on_group_changed: Callable[[List[float]], None]

@xr_slider Decorator
└── Configures slider with min/max, step, haptic
```

### Virtual Keyboard

```
VirtualKeyboard (XRUIPanel)
├── Configuration
│   ├── width: float (default 0.5m)
│   ├── height: float (default 0.25m)
│   ├── key_width: float (per key)
│   ├── key_height: float
│   └── key_spacing: float
├── Layout
│   ├── current_layout: KeyboardLayout
│   ├── available_layouts: List[KeyboardLayout]
│   └── set_layout(layout) -> None
├── State
│   ├── text: str
│   ├── cursor_position: int
│   ├── is_shift_active: bool
│   ├── is_caps_lock: bool
│   └── suggestions: List[str]
├── Key Handling
│   ├── on_key_press(key: KeyInfo) -> None
│   ├── handle_character(char: str) -> None
│   ├── handle_backspace() -> None
│   ├── handle_enter() -> None
│   ├── handle_shift() -> None
│   └── handle_cursor(direction: int) -> None
├── Callbacks
│   ├── on_text_changed: Callable[[str], None]
│   ├── on_submit: Callable[[str], None]
│   └── on_cancel: Callable
└── Hit Testing
    └── get_key_at_uv(uv) -> Optional[KeyInfo]

KeyboardLayout Enum
├── QWERTY
├── QWERTY_UPPER
├── AZERTY
├── QWERTZ
├── NUMERIC
├── SYMBOLS
└── EMOJI

KeyType Enum
├── CHARACTER
├── SHIFT
├── BACKSPACE
├── ENTER
├── SPACE
├── TAB
├── SYMBOLS
├── ABC
├── NUMBERS
├── LANGUAGE
├── HIDE
├── LEFT
├── RIGHT
├── CLEAR
└── EMOJI

KeyInfo
├── type: KeyType
├── character: Optional[str]
├── width: float (relative, 1.0 = normal key)
└── row: int

KeyboardManager (Singleton)
├── show_keyboard(target: TextInput) -> VirtualKeyboard
├── hide_keyboard() -> None
├── is_visible: bool
└── current_target: Optional[TextInput]
```

### Wrist UI

```
WristUI (XRUIPanel)
├── Configuration
│   ├── size: float (meters, from XR_CONFIG)
│   ├── wrist_offset: Vec3 (position offset from wrist)
│   ├── hand: XRHand (LEFT, RIGHT)
│   └── layout: WristUILayout
├── Visibility
│   ├── visibility_mode: WristUIVisibilityMode
│   ├── look_at_threshold: float (dot product)
│   ├── palm_up_threshold: float (degrees)
│   └── is_visible: bool
├── Menu Items
│   ├── items: List[WristMenuItem]
│   ├── add_item(item) -> None
│   ├── remove_item(item) -> None
│   └── max_items: int (8 circular, 12 rectangular)
├── State
│   ├── selected_index: int
│   └── notification_count: Dict[int, int]
├── Tracking Update
│   ├── wrist_position: Vec3
│   ├── wrist_rotation: Quat
│   ├── head_position: Vec3
│   └── head_forward: Vec3
└── Methods
    ├── check_visibility() -> bool
    ├── update_tracking(wrist_pose, head_pose) -> None
    └── select_item(index) -> None

WristUIVisibilityMode Enum
├── ALWAYS
├── LOOK_AT (visible when user looks at wrist)
├── PALM_UP (visible when palm faces up)
└── MANUAL (explicitly toggled)

WristUILayout Enum
├── CIRCULAR (max 8 items, watch face style)
├── RECTANGULAR (max 12 items, 4x3 grid)
└── RADIAL (max 8 items, pie menu)

WristMenuItem
├── id: str
├── icon: Optional[str]
├── label: str
├── on_select: Callable
└── notification_badge: int

WristUIManager (Singleton)
├── left_wrist: Optional[WristUI]
├── right_wrist: Optional[WristUI]
├── create_wrist_ui(hand, layout) -> WristUI
├── update_tracking(left_wrist_pose, right_wrist_pose, head_pose)
└── get_active_wrist() -> Optional[WristUI]
```

## Integration Points

### Dependencies (Incoming)
- Phase 2: Ray origin from controller, poke position from hand, gaze from eye tracking
- Renderer: Panel texture generation, button visual states
- `engine.xr.config`: XR_CONFIG for default sizes

### Dependents (Outgoing)
- Application: Consumes UI events
- Haptics: Receives feedback requests

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Input Sources                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Controller│  │  Hand    │  │   Eye    │                  │
│  │  (Ray)   │  │ (Poke)   │  │ (Gaze)   │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────┐
│                  UI Interaction Manager                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  raycast()     poke()         gaze()                    ││
│  │     │            │              │                        ││
│  │     └────────────┴──────────────┘                        ││
│  │                  │                                        ││
│  │                  ▼                                        ││
│  │     ┌─────────────────────────┐                          ││
│  │     │   Hit Testing           │                          ││
│  │     │   (panel -> element)    │                          ││
│  │     └───────────┬─────────────┘                          ││
│  └─────────────────┼────────────────────────────────────────┘│
└────────────────────┼─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                      UI Elements                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │   Panel   │  │  Button   │  │  Slider   │  │ Keyboard  │ │
│  │           │  │(haptic)   │  │(haptic)   │  │           │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
└────────┼──────────────┼──────────────┼──────────────┼────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Application       │
                    │   (callbacks)       │
                    └─────────────────────┘
```

## Performance Requirements

| Component | Update Rate | CPU Budget |
|-----------|-------------|------------|
| Raycast (per panel) | 90 Hz | <0.02ms |
| Poke Detection | 90 Hz | <0.02ms |
| Gaze Tracking | 90 Hz | <0.01ms |
| Hit Testing | 90 Hz | <0.05ms |
| Keyboard Key Detection | 90 Hz | <0.02ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Poke false positives | High | Medium | Depth threshold tuning, visual feedback |
| Gaze dwell too slow | Medium | Medium | Configurable dwell time, audio feedback |
| Keyboard typing errors | Medium | Medium | Larger keys, haptic per-key feedback |
| Wrist UI visibility flicker | Medium | Low | Hysteresis on visibility threshold |
