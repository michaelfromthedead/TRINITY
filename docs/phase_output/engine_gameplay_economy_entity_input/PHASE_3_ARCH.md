# PHASE 3 ARCHITECTURE: Input Module Testing

## Overview

Comprehensive test coverage for engine/gameplay/input module (~4,064 lines across 4 files).

## Components Under Test

### 1. Device System (devices.py, 1,503 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| DeviceType | Enum (Keyboard, Mouse, Gamepad, Touch, Motion, XR) | Device identification |
| KeyboardDevice | Key states, modifiers, text buffer | State tracking, modifier combos |
| MouseDevice | Position, delta, scroll, sensitivity, capture | Delta calculation, capture toggle |
| GamepadDevice | Axes, triggers, buttons, rumble | Analog values, rumble scheduling |
| TouchDevice | Multi-touch with pressure, phase | Touch lifecycle, pressure range |
| MotionDevice | Gyroscope, accelerometer, orientation | Smoothing, quaternion normalization |
| XRDevice | 6DOF pose, thumbstick, triggers, haptics | Pose transform, haptic patterns |
| DeviceManager | Hot-plug detection, registration | Connect/disconnect events |

### 2. Action Mapper (action_mapper.py, 834 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| TriggerTypes | Pressed, Released, Down, Hold, Tap, DoubleTap, Combo | Enum values |
| TriggerEvaluators | State machines per trigger type | State transitions |
| HoldTrigger | Hold duration tracking | Progress, threshold, timeout |
| TapTrigger | Quick press detection | Max duration |
| DoubleTapTrigger | Two taps within window | Window timing |
| ComboTrigger | Sequence of inputs | Sequence ordering, timeout |
| ActionMapper | Maps inputs to actions | Binding, consumption |
| @input_action | Metadata attachment | Decorator behavior |

### 3. Axis Mapper (axis_mapper.py, 782 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| AxisBindingType | Digital (WASD), Analog (stick), Composite | Type handling |
| AxisMapper | Digital-to-analog conversion | Value mapping |
| Vector2Mapper | 2D axis with dead zone | Radial dead zone, normalization |
| @input_axis | Metadata attachment | Decorator behavior |

### 4. Processing Pipeline (processing.py, 747 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| DeadZoneType | Axial, Radial, Cross | Type handling |
| DeadZone | Dead zone application | Rescaling, zero division |
| ResponseCurve | Linear, Power, Exponential, S-curve, Step | Curve accuracy |
| InputSmoother | Moving average, exponential, double-exponential | Smoothing behavior |
| InputModifierChain | Composable pipeline | Chain ordering |
| InputProcessor | Complete processing | Settings integration |

## Architecture Decisions

### ADR-INP-1: Device Test Fixtures

Create DeviceTestHarness with:
- Mock device creation (no hardware dependency)
- Simulated input injection
- Frame-by-frame state advancement

### ADR-INP-2: Trigger State Machine Testing

Each trigger type must be tested through all states:
- NONE -> STARTED -> ONGOING -> COMPLETED
- NONE -> STARTED -> CANCELLED (early release)

Verify state transitions are deterministic.

### ADR-INP-3: Dead Zone Mathematical Verification

Dead zone tests must verify:
- Input below dead zone -> output is exactly 0
- Input at dead zone boundary -> output is exactly 0
- Input above dead zone -> output is rescaled (not jumped)
- Zero division protection when magnitude = 0

### ADR-INP-4: Response Curve Boundary Values

All response curves must be tested at:
- Input = 0 -> Output = 0
- Input = 1 -> Output = 1
- Input = -1 -> Output = -1 (for signed)
- Midpoint values for S-curve

### ADR-INP-5: Motion Smoothing Verification

Smoothing tests must verify:
- Alpha = 0 -> no smoothing (output = input)
- Alpha = 1 -> infinite smoothing (output = previous)
- Intermediate alpha -> smooth transition

## Test Structure

```
tests/
  input/
    test_device_keyboard.py         # KeyboardDevice
    test_device_mouse.py            # MouseDevice
    test_device_gamepad.py          # GamepadDevice
    test_device_touch.py            # TouchDevice
    test_device_motion.py           # MotionDevice, smoothing
    test_device_xr.py               # XRDevice, 6DOF
    test_device_manager.py          # Hot-plug, registration
    test_trigger_pressed.py         # Pressed trigger
    test_trigger_hold.py            # Hold trigger state machine
    test_trigger_tap.py             # Tap trigger
    test_trigger_doubletap.py       # DoubleTap trigger
    test_trigger_combo.py           # Combo trigger sequence
    test_action_mapper.py           # ActionMapper binding
    test_axis_mapper.py             # AxisMapper, Vector2Mapper
    test_dead_zone_axial.py         # Axial dead zone
    test_dead_zone_radial.py        # Radial dead zone
    test_dead_zone_cross.py         # Cross dead zone
    test_response_curves.py         # All curve types
    test_input_smoother.py          # Smoothing algorithms
    test_modifier_chain.py          # Pipeline composition
```

## Dependencies

- pytest for test framework
- pytest-mock for mocking device hardware
- math module for curve verification

## Risks

| Risk | Mitigation |
|------|------------|
| Floating point precision | Use pytest.approx() for comparisons |
| State machine complexity | Explicit state diagrams in tests |
| Hot-plug timing | Mock all device enumeration |
