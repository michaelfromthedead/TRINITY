# Phase U4: UI Data Binding and Screen Management — Architecture

**Tasks:** T-UX-4.1 through T-UX-4.4 (4 tasks)
**Effort:** 11-15 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase U4 implements data binding (model-to-view synchronization) and screen management (navigation, transitions).

---

## 2. Data Binding (`binding/binding.py`)

### Binding Modes

| Mode | Direction | Descriptor |
|------|-----------|------------|
| OneWay | Model → Widget | ObservableDescriptor |
| TwoWay | Model ↔ Widget | BoundDescriptor |
| OneTime | Model → Widget (once) | ImmutableDescriptor |

### Binding Flow
```
Model Change
    ↓
ObservableDescriptor.notify()
    ↓
Binding.update()
    ↓
Widget.TrackedDescriptor.set()
    ↓
Re-render
```

### Multiple Bindings
Multiple widgets can bind to the same model field. All receive updates.

---

## 3. Value Converters (`binding/converter.py`)

Converters transform values between model and view types.

| Converter | From → To |
|-----------|-----------|
| FloatToPercent | 0.0-1.0 → "0%-100%" |
| FloatToColor | 0.0-1.0 → gradient color |
| IntToString | 42 → "42" |
| DateToString | datetime → "2026-05-25" |

Converters are registered by type pair: `register_converter(float, str, FloatToPercent)`.

---

## 4. Input Validation (`binding/validation.py`)

Validation runs before model update on two-way bindings.

```python
@validate
def validate_positive(value: float) -> bool:
    return value >= 0

# On validation failure:
# - Model is NOT updated
# - Error callback invoked with validation message
# - Widget shows error state
```

---

## 5. Screen Management (`screens/screen.py`)

### Screen Lifecycle
```
@on_enter → Active → @on_exit
```

### Screen Stack
| Operation | Effect |
|-----------|--------|
| push(screen) | Add to top, enter new |
| pop() | Remove top, exit current |
| replace(screen) | Remove top, add new |

### State Machine
`@state_machine` decorator defines valid screen transitions:
```python
@state_machine(transitions={
    "MainMenu": ["Settings", "Game"],
    "Settings": ["MainMenu"],
    "Game": ["PauseMenu", "GameOver"],
})
class GameScreens: ...
```

Invalid navigation raises `InvalidTransitionError`.

---

## 6. Screen Transitions (`screens/transitions.py`)

| Transition | Parameters |
|------------|------------|
| Fade | duration, easing |
| Slide | direction (left/right/up/down), duration |
| Zoom | scale_from, scale_to, duration |

`@tween` decorator configures animation timing.

---

## 7. Dependencies

- Phase U1: Widget base
- Phase U2: Widgets
- Phase U3: Styling
- Trinity: ObservableDescriptor, BoundDescriptor, ValidatedDescriptor
- Foundation: Tracker
