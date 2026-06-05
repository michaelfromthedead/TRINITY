# PHASE 3 TODO: Input Platform Integration

## Summary

Connect input subsystem to platform event sources via SDL2 backend.

**Estimated Effort:** 16-20 hours
**Dependencies:** Phase 1 complete
**Blocking:** None (input is independent)

---

## Tasks

### T-P3-001: Create InputBackend ABC

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Create `engine/platform/input/input_backend.py`:

```python
from abc import ABC, abstractmethod
from typing import Protocol

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

@dataclass
class InputDeviceInfo:
    id: str
    name: str
    device_type: InputDeviceType
    vendor_id: int = 0
    product_id: int = 0
```

**Acceptance Criteria:**
- [ ] ABC defines all required methods
- [ ] InputDeviceInfo captures device metadata
- [ ] Type hints complete

---

### T-P3-002: Create NullInputBackend

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Create `engine/platform/input/backends/null_backend.py`:

```python
class NullInputBackend(InputBackend):
    def __init__(self):
        self._event_queue: list[InputEvent] = []
        self._devices: list[InputDeviceInfo] = []

    def inject(self, event: InputEvent) -> None: ...
    def add_device(self, info: InputDeviceInfo) -> None: ...
    def remove_device(self, device_id: str) -> None: ...
```

**Acceptance Criteria:**
- [ ] Event injection works
- [ ] Device enumeration returns injected devices
- [ ] poll() clears queue after returning
- [ ] set_haptic_effect returns True (no-op)

---

### T-P3-003: Create Backend Registry

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Create `engine/platform/input/backends/__init__.py`:

```python
from engine.platform.registry import BackendRegistry
from .null_backend import NullInputBackend

_registry = BackendRegistry[InputBackend]()
_registry.register("null", NullInputBackend, set_default=True)

def get_backend(name: str | None = None) -> InputBackend:
    return _registry.create(name)
```

**Acceptance Criteria:**
- [ ] Uses generic BackendRegistry from Phase 1
- [ ] Null backend is default
- [ ] get_backend() returns usable backend

---

### T-P3-004: Create Keycode Translation Module

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `engine/platform/input/keycodes.py`:

```python
from .keyboard import KeyCode

# SDL2 keycode mappings
SDL_TO_KEYCODE: dict[int, KeyCode] = {
    # Letters
    0x61: KeyCode.A,  # SDLK_a
    # ... 100+ entries
}

def from_sdl(sdl_key: int) -> KeyCode | None:
    return SDL_TO_KEYCODE.get(sdl_key)

def to_sdl(key: KeyCode) -> int | None:
    # Reverse mapping
    ...
```

**Acceptance Criteria:**
- [ ] All KeyCode enum values have SDL mapping
- [ ] Bidirectional translation works
- [ ] Unknown keys return None

---

### T-P3-005: Create SDL2 Input Backend

**Priority:** P0 (Blocking)
**Estimate:** 6 hours

Create `engine/platform/input/backends/sdl2.py`:

```python
import sdl2
import sdl2.ext

class SDL2InputBackend(InputBackend):
    def initialize(self) -> bool:
        return sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER | sdl2.SDL_INIT_HAPTIC) == 0

    def shutdown(self) -> None:
        sdl2.SDL_Quit()

    def poll(self) -> list[InputEvent]:
        events = []
        for event in sdl2.ext.get_events():
            translated = self._translate_event(event)
            if translated:
                events.append(translated)
        return events

    def _translate_event(self, event: sdl2.SDL_Event) -> InputEvent | None:
        if event.type == sdl2.SDL_KEYDOWN:
            ...
        elif event.type == sdl2.SDL_KEYUP:
            ...
        # Mouse, gamepad, touch...
```

**Acceptance Criteria:**
- [ ] Keyboard events translated correctly
- [ ] Mouse events (move, button, scroll) work
- [ ] Gamepad events (button, axis, trigger) work
- [ ] Touch events work (where SDL2 supports)
- [ ] Device hot-plug detected

---

### T-P3-006: Integrate Backend with InputManager

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Modify `engine/platform/input/input_manager.py`:

```python
class InputManager:
    def __init__(self, backend: InputBackend | None = None):
        if backend is None:
            backend = get_backend()  # Default from registry
        self._backend = backend

    def poll_events(self) -> list[InputEvent]:
        # Poll from backend first
        for event in self._backend.poll():
            self.queue_event(event)
        # Then process queue as before
        ...
```

**Acceptance Criteria:**
- [ ] InputManager accepts backend parameter
- [ ] Default to registry backend
- [ ] Backward compatible (existing code works)
- [ ] Events flow from backend to devices

---

### T-P3-007: Integrate Haptics with Backend

**Priority:** P1 (Important)
**Estimate:** 1.5 hours

Modify `engine/platform/input/haptics.py`:

```python
class Haptics:
    def __init__(self, backend: InputBackend | None = None):
        self._backend = backend

    def play_effect(self, device_id: str, effect: HapticEffect) -> bool:
        if self._backend is None:
            # Queue for later (current behavior)
            self._effect_queue[device_id].append(effect)
            return True
        return self._backend.set_haptic_effect(device_id, effect)
```

**Acceptance Criteria:**
- [ ] Haptics can use backend directly
- [ ] Fallback to queue if no backend
- [ ] Rumble works on gamepad via SDL2

---

### T-P3-008: Implement SDL2 Haptic Support

**Priority:** P1 (Important)
**Estimate:** 2 hours

Add to `backends/sdl2.py`:

```python
def set_haptic_effect(self, device_id: str, effect: HapticEffect) -> bool:
    controller = self._controllers.get(device_id)
    if not controller:
        return False

    if effect.haptic_type == HapticType.RUMBLE:
        return sdl2.SDL_GameControllerRumble(
            controller,
            int(effect.intensity * 65535),
            int(effect.intensity * 65535),
            int(effect.duration * 1000)
        ) == 0
    return False
```

**Acceptance Criteria:**
- [ ] Rumble works on Xbox/PlayStation controllers
- [ ] Unsupported effects return False
- [ ] No crash on missing controller

---

### T-P3-009: Register SDL2 Backend

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Update `backends/__init__.py`:

```python
try:
    from .sdl2 import SDL2InputBackend
    _registry.register("sdl2", SDL2InputBackend)

    # Set as default if available
    if SDL2InputBackend().initialize():
        _registry.register("sdl2", SDL2InputBackend, set_default=True)
except ImportError:
    pass
```

**Acceptance Criteria:**
- [ ] SDL2 backend registered when available
- [ ] Default switches to SDL2 when available
- [ ] ImportError handled gracefully

---

### T-P3-010: Write Backend Tests

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `tests/platform/input/test_backends.py`:

```python
class TestNullInputBackend:
    def test_inject_and_poll(self): ...
    def test_device_enumeration(self): ...
    def test_haptic_effect(self): ...

@pytest.mark.skipif(not SDL2_AVAILABLE, reason="SDL2 not installed")
class TestSDL2InputBackend:
    def test_initialize_shutdown(self): ...
    def test_device_enumeration(self): ...
    # Note: Event tests require display/input, may need to mock
```

**Acceptance Criteria:**
- [ ] Null backend fully tested
- [ ] SDL2 backend tested where possible
- [ ] Tests skip gracefully without SDL2

---

### T-P3-011: Write Keycode Translation Tests

**Priority:** P1 (Important)
**Estimate:** 1 hour

Create `tests/platform/input/test_keycodes.py`:

```python
def test_all_keycodes_mapped():
    for key in KeyCode:
        # Every engine keycode should have SDL mapping
        assert to_sdl(key) is not None

def test_round_trip():
    for key in KeyCode:
        sdl = to_sdl(key)
        back = from_sdl(sdl)
        assert back == key
```

**Acceptance Criteria:**
- [ ] All 100+ keycodes have mappings
- [ ] Round-trip translation works
- [ ] No duplicate mappings

---

## Task Dependency Graph

```
T-P3-001 (InputBackend ABC)
    |
    +-- T-P3-002 (NullInputBackend)
    |       |
    |       +-- T-P3-003 (Registry)
    |               |
    |               +-- T-P3-006 (InputManager Integration)
    |               |
    |               +-- T-P3-009 (Register SDL2)
    |
    +-- T-P3-004 (Keycodes)
    |       |
    |       +-- T-P3-005 (SDL2 Backend)
    |               |
    |               +-- T-P3-008 (SDL2 Haptics)
    |               |
    |               +-- T-P3-009 (Register SDL2)
    |
    +-- T-P3-007 (Haptics Integration)

T-P3-010 (Backend Tests) -- after T-P3-005, T-P3-002
T-P3-011 (Keycode Tests) -- after T-P3-004
```

## Verification Commands

```bash
# Install dependencies
uv pip install pysdl2 pysdl2-dll

# Verify imports
uv run python -c "from engine.platform.input.backends.sdl2 import SDL2InputBackend"

# Run input tests
uv run pytest tests/platform/input/ -v

# Manual test (requires display)
uv run python -c "
from engine.platform.input import InputManager
from engine.platform.input.backends import get_backend
mgr = InputManager(get_backend('sdl2'))
print('Press any key...')
import time; time.sleep(2)
print(mgr.poll_events())
"
```

## Completion Checklist

- [ ] T-P3-001: InputBackend ABC created
- [ ] T-P3-002: NullInputBackend created
- [ ] T-P3-003: Backend registry created
- [ ] T-P3-004: Keycode translation complete
- [ ] T-P3-005: SDL2 backend created
- [ ] T-P3-006: InputManager integrated
- [ ] T-P3-007: Haptics integrated
- [ ] T-P3-008: SDL2 haptics work
- [ ] T-P3-009: SDL2 registered
- [ ] T-P3-010: Backend tests pass
- [ ] T-P3-011: Keycode tests pass
