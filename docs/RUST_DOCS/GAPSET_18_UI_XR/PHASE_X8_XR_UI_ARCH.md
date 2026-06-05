# Phase X8: XR UI — Architecture

**Tasks:** T-XR-8.1 through T-XR-8.6 (6 tasks)
**Effort:** 15-21 days
**Status:** [ ] NOT STARTED (GPU rendering pending)

---

## 1. Overview

Phase X8 implements XR-specific UI: 3D panels, spatial buttons/sliders, virtual keyboard, and wrist-attached UI. Bridges Phase U1-U5 UI framework with XR interaction.

---

## 2. XR UI Panels (`ui/panel.py`)

### XRUIPanel Component
```python
class XRUIPanel:
    # Widget tree to render
    root_widget: TrackedDescriptor[Widget]
    
    # Panel geometry
    width: float  # meters
    height: float  # meters
    pixels_per_meter: int  # 1000 default
    
    # Panel type
    panel_type: PanelType
    # WORLD_SPACE, HEAD_LOCKED, HAND_ATTACHED, WRIST
    
    # Curved panel
    curve_radius: float  # 2m default, 0 = flat
    
    # Billboard
    billboard: TrackedDescriptor[bool]
```

### Rendering
1. Render widget tree to texture
2. Display texture on quad in 3D scene
3. Interact via ray/poke/gaze

---

## 3. XR Button (`ui/button.py`)

### XRButton Component
Extends UI Button with:
- Press depth tracking (physical button feel)
- Haptic feedback on press
- State hooks via `@on_change`

### Interaction Sources
- Ray interactor (aim and click)
- Direct interactor (poke)
- Gaze interactor (dwell)

---

## 4. XR Slider (`ui/slider.py`)

### XRSlider Component
- Draggable thumb in 3D space
- Range value with clamping
- Snap points (optional)

---

## 5. Virtual Keyboard (`ui/keyboard.py`)

### VirtualKeyboard Component
```python
class VirtualKeyboard:
    layout: TrackedDescriptor[KeyboardLayout]
    # QWERTY, NUMERIC, SYMBOLS
    
    text: TrackedDescriptor[str]
```

### Features
- Multi-layout support
- Key press events to focused text input
- IME integration

---

## 6. Wrist UI (`ui/wrist_ui.py`)

### WristUI Component
Panel attached to virtual forearm.

### Visibility
- Appears on wrist turn toward face
- Auto-hide when looking away

### Content
- Quick action buttons
- Notifications
- Mini-map

---

## 7. Decorators

| Decorator | Configuration |
|-----------|---------------|
| `@xr_ui_panel` | panel_type, interaction_mode |
| `@passthrough_layer` | blend_mode, opacity (for AR) |

---

## 8. Decorator Stacks (`trinity/decorators/builtin_stacks/xr.py`)

| Stack | Description |
|-------|-------------|
| `tracked_xr_device` | XR tracking + component |
| `xr_interactable_object` | Interactable + physics |
| `xr_grabbable_object` | Grabbable + haptic |
| `xr_ui_element` | Panel + interaction |
| `teleport_destination` | Teleport area + visual |
| `ar_spatial_anchor` | Anchor + persistence |
| `xr_avatar_component` | Avatar + IK |
| `multiplayer_xr_avatar` | Avatar + network sync |
| `xr_comfort_locomotion` | Locomotion + comfort |
| `full_xr_player` | Complete XR player rig |
| `xr_weapon` | Grabbable + socket + haptic |
| `xr_tool` | Grabbable + interaction |
| `ar_furniture_placement` | Anchor + plane detection |

---

## 9. Dependencies

- Phase U1-U5: UI framework
- Phase X1: XR Runtime
- Phase X3: Render target for panel texture
- Phase X4: Interaction for ray/poke on panels
- Phase X6: Avatar for wrist transform
