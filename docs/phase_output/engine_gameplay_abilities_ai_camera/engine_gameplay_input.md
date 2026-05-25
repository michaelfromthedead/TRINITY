# Investigation: engine/gameplay/input

## Summary
The gameplay input system is a comprehensive, production-quality implementation featuring action mapping with multiple trigger types (Pressed, Released, Hold, Tap, DoubleTap, Combo), axis mapping with digital/analog/composite bindings, multi-device support (Keyboard, Mouse, Gamepad, Touch, Motion, XR), and sophisticated input processing with dead zones, response curves, and smoothing. This is real, functional code with proper state machines, callback systems, and frame-accurate event handling.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 0 | Empty | Package marker only |
| `action_mapper.py` | 835 | Complete | Full action system with 7 trigger types |
| `axis_mapper.py` | 783 | Complete | 1D and 2D axis mapping with smoothing |
| `constants.py` | 198 | Complete | All magic numbers centralized |
| `devices.py` | 1503 | Complete | 6 device types + device manager |
| `processing.py` | 748 | Complete | Dead zones, curves, smoothing, modifiers |

## Input Components

### Action Mapping (`action_mapper.py`)
- **TriggerType**: PRESSED, RELEASED, DOWN, HOLD, TAP, DOUBLE_TAP, COMBO
- **TriggerState**: NONE, STARTED, ONGOING, COMPLETED, CANCELLED
- **TriggerEvaluator**: Base class with full state machine implementations:
  - `PressedTrigger`: Single frame on activation
  - `ReleasedTrigger`: Single frame on deactivation
  - `DownTrigger`: Continuous while held
  - `HoldTrigger`: Triggers after duration threshold
  - `TapTrigger`: Quick press-and-release detection
  - `DoubleTapTrigger`: Double quick-press detection
- **ActionMapper**: Central system managing actions, bindings, callbacks
- **@input_action decorator**: Declarative action binding

### Axis Mapping (`axis_mapper.py`)
- **AxisBindingType**: DIGITAL, ANALOG, COMPOSITE
- **AxisMapper**: Handles positive/negative key bindings with smoothing
- **Vector2Mapper**: 2D axis with radial dead zone and normalization
- **@input_axis decorator**: Declarative axis binding

### Device Abstraction (`devices.py`)
- **DeviceType**: KEYBOARD, MOUSE, GAMEPAD, TOUCH, MOTION, XR
- **KeyboardDevice**: Key states, modifiers, text input buffer
- **MouseDevice**: Position, delta, scroll, buttons, capture mode
- **GamepadDevice**: Dual sticks, triggers, buttons, rumble
- **TouchDevice**: Multi-touch with phases (began/moved/stationary/ended)
- **MotionDevice**: Gyroscope, accelerometer, orientation quaternion
- **XRDevice**: 6DOF pose, thumbstick, trigger, grip, haptics
- **DeviceManager**: Hot-plug detection, connection listeners

### Input Processing (`processing.py`)
- **DeadZoneType**: NONE, AXIAL, RADIAL, CROSS
- **ResponseCurveType**: LINEAR, POWER, EXPONENTIAL, SCURVE, STEP, CUSTOM
- **SmoothingType**: NONE, MOVING_AVERAGE, EXPONENTIAL, DOUBLE_EXPONENTIAL
- **InputSmoother**: Jitter reduction with configurable algorithms
- **InputModifierChain**: Composable processing pipeline
- **InputProcessor**: Complete processing with settings object

## Implementation

- Real action mapping? **YES** - Full state machine implementation with 7 trigger types, proper frame-accurate detection for hold/tap/double-tap/combo patterns
- Real input buffering? **YES** - History-based smoothing, input state tracking per device, text input buffer for keyboards
- Real combo detection? **YES** - DoubleTapTrigger with gap timing, TriggerType.COMBO support, configurable combo windows (DEFAULT_COMBO_WINDOW = 0.5s, DEFAULT_COMBO_INPUT_TIMEOUT = 0.3s)
- Real device support? **YES** - Complete device abstractions for 6 device types with hot-plug detection

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-ready input system. The code quality is high with proper use of slots, dataclasses, type hints, docstrings, and error handling. The architecture follows game engine best practices with clear separation between raw device input, action mapping, and input processing.

## Evidence

### Hold Trigger State Machine (action_mapper.py:168-233)
```python
class HoldTrigger(TriggerEvaluator):
    """Triggers after input is held for a duration."""

    def __init__(self, hold_duration: float = DEFAULT_HOLD_THRESHOLD):
        super().__init__()
        self._hold_duration = hold_duration
        self._hold_time: float = 0.0
        self._triggered = False

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active:
            if self._state == TriggerState.NONE:
                self._state = TriggerState.STARTED
                self._hold_time = 0.0
                self._triggered = False

            self._hold_time += delta_time
            progress = min(1.0, self._hold_time / self._hold_duration)

            if self._hold_time >= self._hold_duration and not self._triggered:
                self._state = TriggerState.COMPLETED
                self._triggered = True
                return TriggerResult(
                    TriggerState.COMPLETED, value,
                    elapsed_time=self._hold_time, progress=1.0
                )
```

### Radial Dead Zone (processing.py:71-101)
```python
def apply_radial_dead_zone(
    x: float,
    y: float,
    dead_zone: float = DEFAULT_RADIAL_DEAD_ZONE,
    outer_zone: float = DEFAULT_OUTER_DEAD_ZONE
) -> Tuple[float, float]:
    """Apply radial dead zone to a 2D input."""
    magnitude = (x * x + y * y) ** 0.5

    if magnitude < dead_zone:
        return (0.0, 0.0)

    if magnitude > outer_zone:
        return (x / magnitude, y / magnitude)

    rescaled_magnitude = (magnitude - dead_zone) / (outer_zone - dead_zone)
    scale = rescaled_magnitude / magnitude

    return (x * scale, y * scale)
```

### Gamepad Device (devices.py:507-700)
```python
class GamepadDevice(InputDeviceBase):
    """Gamepad/controller input device for gameplay."""
    __slots__ = (
        '_axes', '_triggers', '_button_states', '_previous_buttons',
        '_pressed_buttons', '_released_buttons', '_rumble_left',
        '_rumble_right', '_player_index'
    )

    def __init__(self, device_id: str = "gamepad_0", name: str = "Gamepad", player_index: int = 0):
        capabilities = {"axes", "triggers", "buttons", "rumble"}
        super().__init__(device_id, DeviceType.GAMEPAD, name, capabilities)
        self._axes: Dict[str, float] = {
            "left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": 0.0,
        }
        self._triggers: Dict[str, float] = {"left": 0.0, "right": 0.0}
```
