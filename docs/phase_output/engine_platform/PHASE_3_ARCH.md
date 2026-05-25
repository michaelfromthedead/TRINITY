# PHASE 3 ARCHITECTURE: Input Platform Integration

## Phase Overview

Phase 3 connects the input subsystem to platform event sources. The input abstraction layer is complete (1,698 lines of real implementation); this phase adds platform backends to inject events from actual hardware.

## Current State (from Investigation)

| Component | Status | Lines |
|-----------|--------|-------|
| InputManager | REAL | 238 |
| XRController + XRHand | REAL | 349 |
| Keyboard | REAL | 230 |
| Gamepad | REAL | 221 |
| TouchDevice | REAL | 162 |
| Mouse | REAL | 159 |
| Haptics | REAL | 153 |
| PenDevice | REAL | 115 |

**Missing:**
- Platform event sources (SDL, GLFW, winit)
- Haptics output path
- XR runtime integration

## Architectural Decisions

### ADR-P3-001: Platform Event Source Strategy

**Status:** Proposed

**Context:**
Input events come from various sources:
- Win32: GetAsyncKeyState, Raw Input, XInput
- Linux: evdev, libinput
- macOS: IOKit, NSEvent
- Cross-platform: SDL, GLFW

**Decision:**
Use SDL2 via pygame/pysdl2 as primary cross-platform backend:

```python
class SDL2InputBackend(InputBackend):
    def poll(self) -> list[InputEvent]:
        events = []
        for sdl_event in sdl2.ext.get_events():
            if sdl_event.type == sdl2.SDL_KEYDOWN:
                events.append(InputEvent(
                    device_type=InputDeviceType.KEYBOARD,
                    event_type="key_down",
                    data={"key": translate_keycode(sdl_event.key.keysym.sym)}
                ))
            # ... other event types
        return events
```

**Consequences:**
- SDL2 handles platform differences
- Consistent API across desktop platforms
- Gamepad support via SDL2 GameController API
- Touch support on mobile

### ADR-P3-002: Input Backend Interface

**Status:** Proposed

**Context:**
InputManager currently expects events to be injected. Need a way to poll platform events.

**Decision:**
Define InputBackend ABC:

```python
class InputBackend(ABC):
    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def poll(self) -> list[InputEvent]: ...

    @abstractmethod
    def set_haptic_effect(self, device_id: str, effect: HapticEffect) -> bool: ...

    @abstractmethod
    def enumerate_devices(self) -> list[InputDeviceInfo]: ...

class NullInputBackend(InputBackend):
    """Event injection backend for testing."""
    def __init__(self):
        self._event_queue: list[InputEvent] = []

    def inject(self, event: InputEvent) -> None:
        self._event_queue.append(event)

    def poll(self) -> list[InputEvent]:
        events = self._event_queue.copy()
        self._event_queue.clear()
        return events
```

**Consequences:**
- Clean separation between event source and event processing
- Null backend enables testing (current behavior)
- Platform backends plug in via registry

### ADR-P3-003: Haptics Output Integration

**Status:** Proposed

**Context:**
Haptics manager queues effects but has no output path.

**Decision:**
Route haptic effects through InputBackend:

```python
class SDL2InputBackend(InputBackend):
    def set_haptic_effect(self, device_id: str, effect: HapticEffect) -> bool:
        controller = self._controllers.get(device_id)
        if controller is None:
            return False

        if effect.haptic_type == HapticType.RUMBLE:
            # SDL_GameControllerRumble
            return sdl2.SDL_GameControllerRumble(
                controller,
                int(effect.intensity * 65535),  # low frequency
                int(effect.intensity * 65535),  # high frequency
                int(effect.duration * 1000)     # ms
            ) == 0
        return False
```

**Consequences:**
- Haptics work on platforms with SDL2 rumble support
- Advanced haptics (adaptive triggers) require platform-specific code
- Null backend ignores haptic requests (testable)

### ADR-P3-004: XR Input Integration

**Status:** Deferred

**Context:**
XR input requires OpenXR runtime. Complex integration, separate from desktop input.

**Decision:**
Defer XR integration to a future phase focused on XR platform support. Current XRController/XRHand classes remain testable via event injection.

**Consequences:**
- Phase 3 focuses on desktop/mobile input
- XR input remains functional for testing
- XR phase will add OpenXR backend

### ADR-P3-005: Keycode Translation

**Status:** Proposed

**Context:**
Each platform uses different keycode values. Our KeyCode enum is engine-specific.

**Decision:**
Create translation tables per backend:

```python
# In backends/sdl2.py
SDL_TO_ENGINE_KEYCODE = {
    sdl2.SDLK_a: KeyCode.A,
    sdl2.SDLK_b: KeyCode.B,
    # ... 100+ mappings
}

def translate_keycode(sdl_key: int) -> KeyCode | None:
    return SDL_TO_ENGINE_KEYCODE.get(sdl_key)
```

**Consequences:**
- Engine code uses consistent KeyCode enum
- Each backend maintains its own translation table
- Unknown keys return None (logged, not raised)

## Component Diagram

```
engine/platform/input/
    |
    +-- input_manager.py    # InputManager (event processing)
    |
    +-- input_backend.py    # NEW: InputBackend ABC, NullInputBackend
    |
    +-- keyboard.py         # Keyboard device
    +-- mouse.py            # Mouse device
    +-- gamepad.py          # Gamepad device
    +-- touch.py            # TouchDevice
    +-- pen.py              # PenDevice
    +-- haptics.py          # Haptics manager
    +-- xr_input.py         # XR devices (deferred)
    |
    +-- backends/
            |
            +-- __init__.py     # Backend registry
            +-- null_backend.py # Event injection backend
            +-- sdl2.py         # NEW: SDL2 backend
            +-- glfw.py         # FUTURE: GLFW alternative
```

## Data Flow

### Event Poll Flow

```
SDL2InputBackend.poll()
       |
       v
Translate SDL events to InputEvent
       |
       v
InputManager.queue_event(event)
       |
       v
InputManager._event_queue
       |
       v
InputManager.poll_events()
       |
       v
Notify listeners, dispatch to devices
       |
       v
Keyboard.update(events) / Mouse.update(events) / etc.
```

### Haptic Output Flow

```
Haptics.play_effect(device_id, effect)
       |
       v
Haptics._effect_queue[device_id].append(effect)
       |
       v
InputManager.update() [each frame]
       |
       v
InputBackend.set_haptic_effect(device_id, effect)
       |
       v
SDL2: SDL_GameControllerRumble()
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/input/input_backend.py | InputBackend ABC, InputDeviceInfo |
| engine/platform/input/backends/__init__.py | Backend registry |
| engine/platform/input/backends/null_backend.py | Event injection backend |
| engine/platform/input/backends/sdl2.py | SDL2 backend |
| engine/platform/input/keycodes.py | Keycode translation utilities |

### Modified Files

| File | Changes |
|------|---------|
| engine/platform/input/input_manager.py | Accept InputBackend, poll from it |
| engine/platform/input/haptics.py | Route effects to backend |
| engine/platform/input/__init__.py | Export new classes |

## Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pysdl2 | >=0.9.16 | SDL2 bindings |
| pysdl2-dll | >=2.28.0 | SDL2 library (Windows) |

### Native Libraries

| Library | Platforms | Notes |
|---------|-----------|-------|
| SDL2 | All | Installed via system package or pysdl2-dll |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| SDL2 not available | Fall back to null backend (event injection) |
| Keycode mapping incomplete | Log unmapped keys, return None |
| Haptics unsupported | set_haptic_effect returns False |
| Touch not available on desktop | Supported where SDL2 supports it |

## Phase Exit Criteria

1. SDL2InputBackend polls keyboard/mouse/gamepad events correctly
2. Keycode translation covers all KeyCode enum values
3. Haptic rumble works on gamepad
4. Null backend still works for testing
5. All existing tests pass
6. InputManager integration documented
